"""
Utilities for Flash Attention Triton AMD backend.

This module contains essential runtime utilities:
- GPU architecture detection
- Global configuration flags
- Tensor shape/stride helpers
- FP8 type detection
"""
import functools
import os
from dataclasses import dataclass
from typing import Literal, Optional, Union

import torch
import triton


__all__ = [
    # Runtime info
    "get_arch",
    "is_hip",
    # Global config
    "AUTOTUNE",
    "DEBUG",
    "USE_TRITON_ROCM",
    "BWD_MODE",
    "USE_EXP2",
    "PHILOX_SEED",
    "PHILOX_OFFSET",
    "SHAPE_EXPECTATIONS",
    # FP8
    "is_fp8",
    # Shape/stride helpers
    "get_shape_from_layout",
    "get_stride_from_layout",
    "get_padded_headsize",
    # Misc helpers
    "round_multiple",
    # Head grouping (LLC optimization)
    "is_head_grouping_beneficial",
]


# -------------------------------
# GPU Architecture
# -------------------------------
ArchFamily = Literal["cdna", "rdna"]

CDNA_ARCHS = frozenset({"gfx908", "gfx90a", "gfx940", "gfx941", "gfx942", "gfx950"})
RDNA_ARCHS = frozenset({"gfx1030", "gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201"})
FP8_ARCHS = frozenset({"gfx942", "gfx950"})

_RECOMMENDED_FP8_REPLACEMENTS: dict[str, dict[torch.dtype, torch.dtype]] = {
    "gfx942": {
        torch.float8_e4m3fn: torch.float8_e4m3fnuz,
        torch.float8_e5m2: torch.float8_e5m2fnuz,
    },
}

# Infinity Cache (LLC) sizes for AMD GPUs in bytes
# Note: This is the L3/Infinity Cache, NOT the L2 cache
# RDNA3: L2=6MB, Infinity Cache (LLC)=96MB
_LLC_CACHE_SIZES: dict[str, int] = {
    # RDNA2
    "gfx1030": 128 * 1024 * 1024,  # RX 6900 XT - 128 MB Infinity Cache
    # RDNA3 consumer
    "gfx1100": 96 * 1024 * 1024,   # RX 7900 XTX - 96 MB Infinity Cache
    "gfx1101": 64 * 1024 * 1024,   # RX 7800 XT - 64 MB Infinity Cache
    "gfx1102": 32 * 1024 * 1024,   # RX 7600 - 32 MB Infinity Cache
    # RDNA4
    "gfx1200": 32 * 1024 * 1024,   # RX 9060/XT - 32 MB Infinity Cache
    "gfx1201": 64 * 1024 * 1024,   # RX 9070/XT - 64 MB Infinity Cache
}


@dataclass(frozen=True)
class GpuArch:
    """GPU architecture information."""
    name: str  # e.g., "gfx942", "gfx1100"
    family: Optional[ArchFamily] = None

    @property
    def is_cdna(self) -> bool:
        return self.family == "cdna"

    @property
    def is_rdna(self) -> bool:
        return self.family == "rdna"

    @property
    def supports_fp8(self) -> bool:
        """Check if this architecture supports FP8."""
        return self.name in FP8_ARCHS

    def recommended_fp8_dtype(self, dtype: torch.dtype) -> torch.dtype:
        """Get the recommended FP8 dtype for this architecture.
        
        Some architectures prefer different FP8 variants (e.g., fnuz vs fn).
        Returns the input dtype unchanged if no replacement is recommended.
        """
        return _RECOMMENDED_FP8_REPLACEMENTS.get(self.name, {}).get(dtype, dtype)

    @property
    def cu_count(self) -> int:
        """Get the number of compute units on the current GPU."""
        return int(
            torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
        )

    @property
    def llc_size(self) -> Optional[int]:
        """
        Get Infinity Cache (LLC) size in bytes.
        
        For RDNA3, this is the 96 MB Infinity Cache, not the 6 MB L2.
        Returns None for architectures without a known LLC size (e.g., CDNA).
        """
        # Check exact match first
        if self.name in _LLC_CACHE_SIZES:
            return _LLC_CACHE_SIZES[self.name]
        
        # Check prefix match (e.g., gfx1100 matches gfx1100)
        for known_arch, size in _LLC_CACHE_SIZES.items():
            if self.name.startswith(known_arch):
                return size
        
        # No LLC for this architecture
        return None


# -------------------------------
# Global Variables
# -------------------------------
USE_TRITON_ROCM = os.getenv("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE") == "TRUE"
AUTOTUNE = os.environ.get("FLASH_ATTENTION_TRITON_AMD_AUTOTUNE", "0").lower() in (
    "1",
    "true",
    "yes",
)

# Unified debug level:
#   0 = off (default)
#   1 = basic debug info (shapes, tensor stats, kernel params)
#   2 = detailed debug (includes Triton interpreter prints in kernels)
#
# Set via: FLASH_ATTENTION_TRITON_AMD_DEBUG=0|1|2
DEBUG: int = int(os.environ.get("FLASH_ATTENTION_TRITON_AMD_DEBUG", "0"))
DISABLE_HEAD_GROUPING = os.environ.get("FLASH_ATTN_DISABLE_HEAD_GROUPING", "0") == "1"

if AUTOTUNE or DEBUG > 0:
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
if DEBUG >= 2:
    os.environ["TRITON_INTERPRET"] = "1"
BWD_MODE: Literal["fused", "fused_atomic", "split"] = "fused"
USE_EXP2 = True
PHILOX_SEED = 0x1BF58
PHILOX_OFFSET = 0x1D4B49
SHAPE_EXPECTATIONS: Literal["exact", "rounded"] = "exact"


# -------------------------------
# FP8
# -------------------------------
_FP8_DTYPES = frozenset({
    torch.float8_e4m3fnuz,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
})


def is_fp8(
    x: Union[torch.dtype, torch.Tensor, list[torch.Tensor], tuple[torch.Tensor, ...]],
) -> bool:
    """Check if dtype/tensor(s) are FP8.

    This is a pure function - it only checks dtypes, not architecture support.
    Use `get_arch().supports_fp8` to check if the current GPU supports FP8.

    Args:
        x: A dtype, tensor, or list/tuple of tensors to check.

    Returns:
        True if FP8, False otherwise.

    Rules for multiple tensors:
        - If all tensors are FP8 -> return True.
        - If none are FP8 -> return False.
        - If a mix of FP8 and non-FP8 -> raise ValueError.

    Empty list/tuple returns False.
    """
    # Handle dtype directly
    if isinstance(x, torch.dtype):
        return x in _FP8_DTYPES

    # Handle single tensor
    if isinstance(x, torch.Tensor):
        return x.dtype in _FP8_DTYPES

    # Handle list/tuple of tensors
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return False
        flags = [t.dtype in _FP8_DTYPES for t in x]
        if all(flags):
            return True
        if not any(flags):
            return False
        raise ValueError(
            "Mixed FP8 and non-FP8 tensors provided; either all or none must be FP8."
        )

    raise TypeError(f"Expected dtype, Tensor, or sequence of Tensors, got {type(x)}")


# -------------------------------
# Shape/Stride Helpers
# -------------------------------
def get_shape_from_layout(
    x: torch.Tensor,
    layout: Literal["bshd", "bhsd", "thd"],
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
) -> tuple[int, int, int, int]:
    """Extract (batch, max_seqlen, num_heads, head_dim) from tensor based on layout."""
    if layout == "bhsd":
        batch, num_heads, max_seqlen_final, head_dim = x.shape
    elif layout == "bshd":
        batch, max_seqlen_final, num_heads, head_dim = x.shape
    elif layout == "thd":
        total_seqlen, num_heads, head_dim = x.shape
        if cu_seqlens is None:
            raise ValueError("cu_seqlens must be provided for varlen (thd) layout")
        if max_seqlen is None:
            raise ValueError("max_seqlen must be provided for varlen (thd) layout")

        batch, max_seqlen_final, num_heads, head_dim = (
            len(cu_seqlens) - 1,
            max_seqlen,
            num_heads,
            head_dim,
        )
    else:
        raise ValueError(f"Got unsupported layout: {layout}")

    return batch, max_seqlen_final, num_heads, head_dim


def get_stride_from_layout(
    x: torch.Tensor, layout: Literal["bshd", "bhsd", "thd"]
) -> tuple[int, int, int, int]:
    """Get strides in (batch, head, seq, dim) order for the given layout."""
    if layout == "thd":
        strides = (0, x.stride(1), x.stride(0), x.stride(2))
    elif layout == "bhsd":
        strides = (x.stride(0), x.stride(1), x.stride(2), x.stride(3))
    elif layout == "bshd":
        strides = (x.stride(0), x.stride(2), x.stride(1), x.stride(3))
    else:
        raise ValueError(f"Got unsupported layout: {layout}")
    return strides


def get_padded_headsize(size: int) -> int:
    """Get closest power of 2 over or equal to 32."""
    # Smallest head_dim supported is 16. If smaller, the tile in the
    # kernel is padded - there is no padding in memory for any dims.
    padded_d_model = 1 << (size - 1).bit_length()
    padded_d_model = max(padded_d_model, 16)
    return padded_d_model


# -------------------------------
# Misc helpers
# -------------------------------
def round_multiple(x: int, m: int) -> int:
    """Round x up to the nearest multiple of m."""
    return (x + m - 1) // m * m


def _get_dtype_size(dtype: torch.dtype) -> Optional[int]:
    """Get element size in bytes for a dtype. Returns None if unknown."""
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    elif dtype == torch.float32:
        return 4
    elif 'float8' in str(dtype).lower():
        return 1
    return None


def calculate_optimal_head_group_size(
    seqlen_k: int,
    head_dim: int,
    dtype: torch.dtype,
    llc_size: int,
    llc_utilization: float = 1.0,
) -> int:
    """
    Calculate the optimal number of heads to process together to fit K,V in LLC.
    """
    elem_size = _get_dtype_size(dtype) or 2  # Default to fp16
    
    # Memory for K and V per head
    kv_per_head = seqlen_k * head_dim * elem_size * 2  # *2 for K and V
    
    # Target LLC usage
    target_llc = int(llc_size * llc_utilization)
    
    # Calculate number of heads that fit
    if kv_per_head == 0:
        return 1
    
    return max(1, target_llc // kv_per_head)


def is_head_grouping_beneficial(
    nheads: int,
    seqlen_k: int,
    head_dim: int,
    dtype: torch.dtype,
    threshold_ratio: float = 1.5,
) -> tuple[bool, int]:
    """
    Determine if head grouping would be beneficial and return optimal group size.
    
    Currently limited to RDNA GPUs with known LLC sizes.
    
    Returns:
        (should_group, group_size): If should_group is False, group_size equals nheads.
    """
    # Check if disabled via environment
    if DISABLE_HEAD_GROUPING:
        return False, nheads
    
    # Only apply head grouping to RDNA GPUs with known LLC
    arch = get_arch()
    llc_size = arch.llc_size
    if llc_size is None:
        return False, nheads
    
    elem_size = _get_dtype_size(dtype) or 2  # Default to fp16
    
    # Total K,V memory for all heads
    total_kv = nheads * seqlen_k * head_dim * elem_size * 2
    
    # Only group if K,V significantly exceeds LLC
    if total_kv < llc_size * threshold_ratio:
        return False, nheads
    
    # Calculate optimal group size
    group_size = calculate_optimal_head_group_size(
        seqlen_k, head_dim, dtype, llc_size
    )
    
    # Only group if we'd have at least 2 groups
    if group_size >= nheads:
        return False, nheads
    
    # Minimum group size to avoid excessive kernel launches
    min_group_size = max(1, nheads // 16)  # At most 16 groups
    group_size = max(group_size, min_group_size)
    
    return True, min(group_size, nheads)


# -------------------------------
# Runtime info
# -------------------------------
@functools.cache
def is_hip() -> bool:
    """Check if running on HIP (AMD) backend."""
    return bool(triton.runtime.driver.active.get_current_target().backend == "hip")


@functools.cache
def get_arch() -> GpuArch:
    """Get the current GPU architecture."""
    name: str = triton.runtime.driver.active.get_current_target().arch
    if name in CDNA_ARCHS:
        return GpuArch(name=name, family="cdna")
    elif name in RDNA_ARCHS:
        return GpuArch(name=name, family="rdna")
    else:
        return GpuArch(name=name)
