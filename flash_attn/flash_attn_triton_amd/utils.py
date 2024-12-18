
import csv
import json
import math
import torch
import os
import random
import triton
import triton.language as tl

AUTOTUNE = os.environ.get('FLASH_ATTENTION_TRITON_AMD_AUTOTUNE', '0').lower() in ('1', 'true', 'yes')
DEBUG = os.environ.get('FLASH_ATTENTION_TRITON_AMD_DEBUG', '0').lower() in ('1', 'true', 'yes')
PERF = os.environ.get('FLASH_ATTENTION_TRITON_AMD_PERF', '0').lower() in ('1', 'true', 'yes')
REMOVE_QUANTIZATION_SCALING = os.environ.get('FLASH_ATTENTION_TRITON_AMD_REMOVE_QUANT_SCALE', '0').lower() in ('1', 'true', 'yes')
USE_TRITON_ROCM = os.getenv("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE") == "TRUE"
if USE_TRITON_ROCM: # TODO remove this
    random.seed(42)
DROPOUT_USE_PYTORCH = False
DROPOUT_DUMP = False

class MetaData():
    cu_seqlens_q = None
    cu_seqlens_k = None
    max_seqlens_q = 0
    max_seqlens_k = 0
    bias = None
    alibi_slopes = None
    causal = False
    num_contexts = 0
    varlen = False
    layout = None
    cache_seqlens = None
    cache_batch_idx = None
    new_kv = False
    seqlen_new = None
    k_new = None
    v_new = None
    return_scores= False
    dropout_p= 0.0
    philox_seed, philox_offset = None, None # if dropout_p > 0.0 seed the RNG so we get reproducible results for testing.
    # NOTE: scale sm_scale by log_2(e) and use 2^x in the loop as we do not have native e^x support in HW.
    use_exp2 = False
    rotary_sin = None
    rotary_cos = None
    rotary_interleaved = False
    rotary_conjunction = False
    

    def __repr__(self) -> str:
        return (f"MetaData(\n"
                f"  sm_scale={self.sm_scale},\n"
                f"  cu_seqlens_q={self.cu_seqlens_q},\n"
                f"  cu_seqlens_k={self.cu_seqlens_k},\n"
                f"  max_seqlens_q={self.max_seqlens_q},\n"
                f"  max_seqlens_k={self.max_seqlens_k},\n"
                f"  bias={self.bias},\n"
                f"  alibi_slopes={self.alibi_slopes},\n"
                f"  causal={self.causal},\n"
                f"  num_contexts={self.num_contexts},\n"
                f"  varlen={self.varlen},\n"
                f"  layout={self.layout},\n"
                f"  cache_seqlens={self.cache_seqlens},\n"
                f"  cache_batch_idx={self.cache_batch_idx},\n"
                f"  new_kv={self.new_kv},\n"
                f"  seqlen_new={self.seqlen_new},\n"
                f"  k_new={self.k_new},\n"
                f"  v_new={self.v_new},\n"
                f"  dropout_p={self.dropout_p},\n"
                f"  return_scores={self.return_scores}\n"
                f")")

    def __init__(self, sm_scale=1.0):
        self.sm_scale = sm_scale

    def set_varlen_params(self, cu_seqlens_q, cu_seqlens_k):
        self.varlen = True
        self.layout = 'thd'
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        # Without "varlen", there should still be one sequence.
        assert len(cu_seqlens_q) >= 2
        assert len(cu_seqlens_q) == len(cu_seqlens_k)
        self.num_contexts = len(cu_seqlens_q) - 1
        for i in range(0, self.num_contexts):
            self.max_seqlens_q = max(cu_seqlens_q[i + 1].item() - cu_seqlens_q[i].item(), self.max_seqlens_q)
            self.max_seqlens_k = max(cu_seqlens_k[i + 1].item() - cu_seqlens_k[i].item(), self.max_seqlens_k)

    def need_bias(self, bias, batch, nheads, seqlen_q, seqlen_k):
        assert bias.is_cuda
        assert bias.dim() == 4
        assert bias.shape[0] == 1
        assert bias.shape[2:] == (seqlen_q, seqlen_k)
        self.bias = bias

    def need_alibi(self, alibi_slopes, batch, nheads):
        assert alibi_slopes.is_cuda
        assert alibi_slopes.dim() == 2
        assert alibi_slopes.shape[0] == batch
        assert alibi_slopes.shape[1] == nheads
        self.alibi_slopes = alibi_slopes

    def need_causal(self):
        self.causal = True

    def need_rotary(self, sin, cos, rotary_interleaved, rotary_conjunction=False):
        self.rotary_sin = sin
        self.rotary_cos = cos
        self.rotary_interleaved = rotary_interleaved
        self.rotary_conjunction = rotary_conjunction

    def need_dropout(self, dropout_p):
        self.dropout_p = dropout_p
        self.return_scores = True
        self.philox_seed, self.philox_offset = 0x1BF58, 0x1D4B49

    def check_args(self, q, k, v, o):
        assert q.dim() == k.dim() and q.dim() == v.dim()

        batch, nheads_q, nheads_k, head_size, _, _ = get_shape_from_layout(q, k, self.layout, self.cu_seqlens_q, self.cu_seqlens_k, self.max_seqlens_q, self.max_seqlens_k)
        if self.varlen:
            assert q.dim() == 3
            assert self.cu_seqlens_q is not None
            assert self.cu_seqlens_k is not None
            assert len(self.cu_seqlens_q) == len(self.cu_seqlens_k)
            # TODO: Remove once bias is supported with varlen
            assert self.bias is None
            # assert not self.return_scores
        else:
            assert q.dim() == 4
            assert self.max_seqlens_q > 0 and self.max_seqlens_k > 0
            assert self.cu_seqlens_q is None and self.cu_seqlens_k is None
        assert k.shape == v.shape
        assert q.shape[-1] == k.shape[-1] and q.shape[-1] == v.shape[-1]
        # TODO: Change assert if we support qkl f8 and v f16
        assert q.dtype == k.dtype and q.dtype == v.dtype
        assert head_size <= 256
        assert o.shape == q.shape
        assert (nheads_q % nheads_k) == 0
        assert self.layout is not None
        assert self.layout == 'thd' or not self.varlen

def input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout, device="cuda", DEBUG_INPUT=False):
    torch.manual_seed(20)

    # Initialize q, k, v
    if layout == 'bhsd':
        q_tensor_shape = (Z, HQ, N_CTX_Q, D_HEAD)
        k_tensor_shape = (Z, HK, N_CTX_K, D_HEAD)
    elif layout == 'bshd':
        q_tensor_shape = (Z, N_CTX_Q, HQ, D_HEAD)
        k_tensor_shape = (Z, N_CTX_K, HK, D_HEAD)
    else:
        assert False, f'Got unsupported tensor layout: {layout}'

    if DEBUG_INPUT:
        if layout == "bhsd":
            q = torch.arange(N_CTX_Q, dtype=torch.float32, device=device).view(1, 1, N_CTX_Q, 1).expand(*q_tensor_shape).contiguous().requires_grad_()
            k = torch.arange(N_CTX_K, dtype=torch.float32, device=device).view(1, 1, N_CTX_K, 1).expand(*k_tensor_shape).contiguous().requires_grad_()
            v = torch.arange(N_CTX_K, dtype=torch.float32, device=device).view(1, 1, N_CTX_K, 1).expand(*k_tensor_shape).contiguous().requires_grad_()
        elif layout == "bshd":
            q = torch.arange(N_CTX_Q, dtype=torch.float32, device=device).view(1, N_CTX_Q, 1, 1).expand(*q_tensor_shape).contiguous().requires_grad_()
            k = torch.arange(N_CTX_K, dtype=torch.float32, device=device).view(1, N_CTX_K, 1, 1).expand(*k_tensor_shape).contiguous().requires_grad_()
            v = torch.arange(N_CTX_K, dtype=torch.float32, device=device).view(1, N_CTX_K, 1, 1).expand(*k_tensor_shape).contiguous().requires_grad_()
    else:
        q = torch.randn(q_tensor_shape, dtype=torch.float32, device=device, requires_grad=True)
        k = torch.randn(k_tensor_shape, dtype=torch.float32, device=device, requires_grad=True)
        v = torch.randn(k_tensor_shape, dtype=torch.float32, device=device, requires_grad=True)

    q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    
    if DEBUG_INPUT:
        sm_scale = 1
    else:
        sm_scale = D_HEAD**-0.5
    input_metadata = MetaData(sm_scale=sm_scale)
    input_metadata.max_seqlens_q = N_CTX_Q
    input_metadata.max_seqlens_k = N_CTX_K
    input_metadata.layout = layout
    return q, k, v, input_metadata


def varlen_input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, device="cuda", equal_seqlens=False, DEBUG_INPUT=False):
    torch.manual_seed(20)

    # Random or equal sequence lengths based on 'equal_seqlens' flag
    if not equal_seqlens:
        max_seqlens_q = N_CTX_Q // Z
        max_seqlens_k = N_CTX_K // Z
        seqlens_q = torch.randint(1, max_seqlens_q + 1, (Z,), dtype=torch.int32)
        seqlens_k = torch.randint(1, max_seqlens_k + 1, (Z,), dtype=torch.int32)
    else:
        seqlens_q = torch.full((Z,), N_CTX_Q // Z, dtype=torch.int32)
        seqlens_k = torch.full((Z,), N_CTX_K // Z, dtype=torch.int32)

    # Calculate cumulative sequence lengths
    cu_seqlens_q = torch.cat([torch.tensor([0], dtype=torch.int32), seqlens_q.cumsum(dim=0)])
    cu_seqlens_k = torch.cat([torch.tensor([0], dtype=torch.int32), seqlens_k.cumsum(dim=0)])
    cu_seqlens_q = cu_seqlens_q.to(device=device).to(torch.int32)
    cu_seqlens_k = cu_seqlens_k.to(device=device).to(torch.int32)

    # Total lengths
    total_q = cu_seqlens_q[-1].item()
    total_k = cu_seqlens_k[-1].item()

    if DEBUG_INPUT:
        # Initialize q, k, v with deterministic values
        q = torch.arange(total_q, dtype=torch.float32, device=device).view(total_q, 1, 1)
        q = q.expand(total_q, HQ, D_HEAD).contiguous().requires_grad_()
        k = torch.arange(total_k, dtype=torch.float32, device=device).view(total_k, 1, 1)
        k = k.expand(total_k, HK, D_HEAD).contiguous().requires_grad_()
        v = torch.arange(total_k, dtype=torch.float32, device=device).view(total_k, 1, 1)
        v = v.expand(total_k, HK, D_HEAD).contiguous().requires_grad_()
        sm_scale = 1
    else:
        # Initialize q, k, v with random values
        q = torch.randn((total_q, HQ, D_HEAD), dtype=torch.float32, device=device).requires_grad_()
        k = torch.randn((total_k, HK, D_HEAD), dtype=torch.float32, device=device).requires_grad_()
        v = torch.randn((total_k, HK, D_HEAD), dtype=torch.float32, device=device).requires_grad_()
        sm_scale = D_HEAD ** -0.5

    q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)

    input_metadata = MetaData(sm_scale=sm_scale)
    input_metadata.set_varlen_params(cu_seqlens_q, cu_seqlens_k)
    return q, k, v, input_metadata


def get_shape_from_layout(q, k, layout, cu_seqlens_q = None, cu_seqlens_k = None, max_seqlen_q=None, max_seqlen_k=None):
    if layout == 'bhsd':
        batch_q, nheads_q, max_seqlen_q, head_size_q = q.shape
        batch_k, nheads_k, max_seqlen_k, head_size_k = k.shape
    elif layout == 'bshd':
        batch_q, max_seqlen_q, nheads_q, head_size_q = q.shape
        batch_k, max_seqlen_k, nheads_k, head_size_k = k.shape
    elif  layout == 'thd':
        batch_q, max_seqlen_q, nheads_q, head_size_q = len(cu_seqlens_q) - 1, max_seqlen_q, q.shape[1], q.shape[2]
        batch_k, max_seqlen_k, nheads_k, head_size_k = len(cu_seqlens_k) - 1, max_seqlen_k, k.shape[1], k.shape[2]
    else:
        assert False, "Got unsupported layout."
    
    # assert
    assert batch_q == batch_k
    assert head_size_q == head_size_k

    return batch_q, nheads_q, nheads_k, head_size_q, max_seqlen_q, max_seqlen_k

def get_strides_from_layout(q, k, v, o, layout):
    if layout == 'thd':
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
        v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
        o_strides = (0, o.stride(1), o.stride(0), o.stride(2))
    elif layout == 'bhsd':
        q_strides = (q.stride(0), q.stride(1), q.stride(2), q.stride(3))
        k_strides = (k.stride(0), k.stride(1), k.stride(2), k.stride(3))
        v_strides = (v.stride(0), v.stride(1), v.stride(2), v.stride(3))
        o_strides = (o.stride(0), o.stride(1), o.stride(2), o.stride(3))
    elif layout == 'bshd':
        q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
        k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
        v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
        o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))
    else:
        assert False, 'Got unsupported layout.'
    return q_strides, k_strides, v_strides, o_strides

def get_padded_headsize(size):
    # Get closest power of 2 over or equal to 32.
    padded_d_model = 1 << (size - 1).bit_length()
    # Smallest head_dim supported is 16. If smaller, the tile in the
    # kernel is padded - there is no padding in memory for any dims.
    padded_d_model = max(padded_d_model, 16)
    return padded_d_model

def compute_alibi_tensor_ref(alibi_slopes, seqlen_q, seqlen_k):
    q_idx = torch.arange(seqlen_q, dtype=torch.int32, device="cuda").unsqueeze(-1)  # (N_CTX_Q, 1)
    k_idx = torch.arange(seqlen_k, dtype=torch.int32, device="cuda").unsqueeze(0)  # (1, N_CTX_K)
    relative_pos = torch.abs(q_idx + seqlen_k - seqlen_q - k_idx)  # (N_CTX_Q, N_CTX_K)
    return -1 * alibi_slopes.unsqueeze(-1).unsqueeze(-1) * relative_pos  # (Z, H, N_CTX_Q, N_CTX_K)

def create_dropout_mask(dropout_p, shape, seed):
    device = "cuda"
    rand_vals = torch.rand(shape, generator=torch.Generator(device=device).manual_seed(seed), device=device, dtype=torch.float32)
    return rand_vals > dropout_p

def write_dropout_mask(x, tensor_name = "tensor"):
    batch, head, seqlen_m, seqlen_n = x.shape
    x = x.tolist()

    with open(f'{tensor_name}.csv', 'w') as f:
        writer = csv.writer(f)
        for b in range(batch):
            for h in range(head):
                dropout_mask = x[b][h]
                if True:
                    BLOCK_M = 64
                    BLOCK_N = 64
                
                    # Calculate number of blocks in each dimension
                    m_blocks = math.ceil(seqlen_m / BLOCK_M)
                    n_blocks = math.ceil(seqlen_n / BLOCK_N)
                    
                    # Process each block
                    for m_block in range(m_blocks):
                        # Calculate row range for current block
                        row_start = m_block * BLOCK_M
                        row_end = min(row_start + BLOCK_M, seqlen_m)
                        
                        for n_block in range(n_blocks):
                            # Calculate column range for current block
                            col_start = n_block * BLOCK_N
                            col_end = min(col_start + BLOCK_N, seqlen_n)
                            
                            # Extract and write the current block
                            for row_idx in range(row_start, row_end):
                                row_data = dropout_mask[row_idx][col_start:col_end]
                                writer.writerow(row_data)
                else:
                    writer.writerows(dropout_mask)

def _strides(x: torch.Tensor, *stride_names: str):
    if x is None:
        return {f"stride_{s}": 0 for i, s in enumerate(stride_names)}

    assert x.ndim == len(stride_names)
    return {f"stride_{s}": x.stride(i) for i, s in enumerate(stride_names)}

def get_input_shapes():
    cases = [(max(1, 2**(16 - i)), 1, 2**i, 16, 1, 128)
             for i in range(8, 18)] + [(max(1, 2**(16 - i)), 1, 2**i, 16, 2, 128) for i in range(8, 18)]
    return cases

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

def get_arch():
    return triton.runtime.driver.active.get_current_target().arch

def is_cdna():
    return is_hip() and get_arch() in ('gfx940', 'gfx941', 'gfx942', 'gfx90a', 'gfx908')


def is_rdna():
    return is_hip() and get_arch() in ("gfx1030", "gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201")


def check_is_fp8(x: torch.Tensor):
    if REMOVE_QUANTIZATION_SCALING:
        return False # makes all methods believe they aren't working with fp8s, so no scaling is applied
    
    fp8_types = {
        torch.float8_e4m3fnuz,
        torch.float8_e4m3fn,  
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    }
    return x.dtype in fp8_types


def create_scale_tensors(q, k, v, SCALE_PER_HEAD=False, layout='bshd', cu_seqlens_q=None, cu_seqlens_k=None):
    """
    Create scale tensors for q and k based on the scaling configuration.
    
    Args:
    q (torch.Tensor): Query tensor
    k (torch.Tensor): Key tensor
    v (torch.Tensor): Value tensor
    SCALE_PER_HEAD (bool): Whether to compute scale per head or globally
    
    Returns:
    tuple: (q_scale, k_scale, v_scale) tensors
    """
    is_fp8 = check_is_fp8(q)

    if layout == 'bhsd':
        seqlen_loc = 2
        dim_loc = 3
    elif layout == 'bshd':
        seqlen_loc = 1
        dim_loc = 3
    else:
        # is varlen
        pass

    is_varlen = layout == "thd"

    # Handle float8 dtype special case:
    if is_fp8:
        # Convert to float32 for scale computation.
        q_float32 = q.to(torch.float32)
        k_float32 = k.to(torch.float32)
        v_float32 = v.to(torch.float32)

        if SCALE_PER_HEAD:
            if is_varlen:
                # FIXME: varlen should be supported.
                assert False, "VARLEN NOT SUPPORTED FOR SCALE PER HEAD"
            else:
                # Compute max for each batch-head pair.
                # Compute max across seqlen and dim.
                q_scale = q_float32.abs().amax(dim=(seqlen_loc, dim_loc))  # Shape: (BATCH, HEAD)
                k_scale = k_float32.abs().amax(dim=(seqlen_loc, dim_loc))  # Shape: (BATCH, HEAD)
                v_scale = v_float32.abs().amax(dim=(seqlen_loc, dim_loc))  # Shape: (BATCH, HEAD)
        else:
            # Compute global max and create a tensor of that value.
            q_global_max = q_float32.abs().max().item()
            k_global_max = k_float32.abs().max().item()
            v_global_max = v_float32.abs().max().item()

            # Create tensors filled with the global max.
            if layout == "bshd":
                batch_q, _, head_q, _ = q.shape
                batch_k, _, head_k, _ = k.shape
            elif layout == "bhsd":
                batch_q, head_q, _, _ = q.shape
                batch_k, head_k, _, _ = k.shape
            elif layout == "thd":
                assert cu_seqlens_q is not None
                batch_q = len(cu_seqlens_q) - 1
                head_q = q.shape[1]
                assert cu_seqlens_k is not None
                batch_k = len(cu_seqlens_k) - 1
                head_k = k.shape[1]
            assert batch_q == batch_k
            q_scale = torch.full((batch_q, head_q), q_global_max, device=q.device)
            k_scale = torch.full((batch_k, head_k), k_global_max, device=k.device)
            v_scale = torch.full((batch_k, head_k), v_global_max, device=v.device)

        # Divide max tensors by respective data type max.
        dtype_max = {
            dtype: torch.finfo(dtype).max
            for dtype in [
                torch.float8_e5m2,
                torch.float8_e5m2fnuz,
                torch.float8_e4m3fn,
                torch.float8_e4m3fnuz,
            ]
        }
        q_scale = q_scale / dtype_max[q.dtype]
        k_scale = k_scale / dtype_max[k.dtype]
        v_scale = v_scale / dtype_max[v.dtype]
    else:
        # For non-float8 dtypes, use a default scale of 1.
        if layout == 'bshd':
            batch, _, head, _ = q.shape
        elif layout == 'bhsd':
            batch, head, _, _ = q.shape
        else:
            # FIXME: varlen should be supported.
            assert False, "VARLEN NOT SUPPORTED"
        q_scale = torch.ones((batch, head), device=q.device)
        k_scale = torch.ones((batch, head), device=k.device)
        v_scale = torch.ones((batch, head), device=v.device)
    
    return q_scale, k_scale, v_scale
