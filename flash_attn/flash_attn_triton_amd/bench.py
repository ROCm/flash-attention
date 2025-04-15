import argparse
import os
import sys
import torch
import triton
import time
import pandas as pd
from logging import warning
from typing import Dict, List, Literal, Optional, Tuple
from functools import lru_cache
from utils import get_arch, input_helper

DEBUG = False

ENV_FLAGS = ["FLASH_ATTENTION_TRITON_AMD_ENABLE", "FLASH_ATTENTION_TRITON_AMD_AUTOTUNE", "FLASH_ATTENTION_TRITON_AMD_DEBUG"]

FUNCTIONS = [
    "flash_attn_func",
    "flash_attn_fp8_func",
    "flash_attn_kvpacked_func",
    "flash_attn_qkvpacked_func",
    "flash_attn_qkvpacked_fp8_func",
    "flash_attn_varlen_func",
    "flash_attn_varlen_fp8_func",
    "flash_attn_varlen_kvpacked_func",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_varlen_qkvpacked_fp8_func",
    "flash_attn_with_kvcache",
]

SUPPORTED_DTYPES = {
    "flash_attn_func": [torch.float16],
    "flash_attn_fp8_func": [torch.float8_e4m3fnuz],
    "flash_attn_kvpacked_func": [torch.float16],
    "flash_attn_qkvpacked_func": [torch.float16],
    "flash_attn_qkvpacked_fp8_func": [torch.float16],
    "flash_attn_varlen_func": [torch.float16],
    "flash_attn_varlen_fp8_func": [torch.float8_e4m3fnuz],
    "flash_attn_varlen_kvpacked_func": [torch.float16],
    "flash_attn_varlen_qkvpacked_func": [torch.float16],
    "flash_attn_varlen_qkvpacked_fp8_func": [torch.float16],
    "flash_attn_with_kvcache": [torch.float16],
}

SUPPORTED_BACKENDS = {
    "flash_attn_func": ["ck", "triton"],
    "flash_attn_fp8_func": ["triton"],
    "flash_attn_kvpacked_func": ["ck", "triton"],
    "flash_attn_qkvpacked_func": ["ck", "triton"],
    "flash_attn_qkvpacked_fp8_func": ["triton"],
    "flash_attn_varlen_func": ["ck", "triton"],
    "flash_attn_varlen_fp8_func": ["triton"],
    "flash_attn_varlen_kvpacked_func": ["ck", "triton"],
    "flash_attn_varlen_qkvpacked_func": ["ck", "triton"],
    "flash_attn_varlen_qkvpacked_fp8_func": ["triton"],
    "flash_attn_with_kvcache": ["ck", "triton"],
}

SUPPORTED_MODES = {
    "flash_attn_func": ["fwd", "bwd", "full"],
    "flash_attn_fp8_func": ["fwd", "bwd", "full"],
    "flash_attn_kvpacked_func": ["fwd", "bwd", "full"],
    "flash_attn_qkvpacked_func": ["fwd", "bwd", "full"],
    "flash_attn_qkvpacked_fp8_func": ["fwd", "bwd", "full"],
    "flash_attn_varlen_func": ["fwd", "bwd", "full"],
    "flash_attn_varlen_fp8_func": ["fwd", "bwd", "full"],
    "flash_attn_varlen_kvpacked_func": ["fwd", "bwd", "full"],
    "flash_attn_varlen_qkvpacked_func": ["fwd", "bwd", "full"],
    "flash_attn_varlen_qkvpacked_fp8_func": ["fwd", "bwd", "full"],
    "flash_attn_with_kvcache": ["fwd"],
}

@lru_cache()
def available_backends():
    available = []
    
    # try to load each backend
    for backend in ["triton", "ck"]:
        try:
            # try loading the module with this backend
            flash_attn = load_flash_attn_module(backend)
            
            # if we got here, the backend loaded successfully
            available.append(backend)
        except Exception as e:
            # backend not available, just continue
            print(f"Backend {backend} not available. Error: {e}")
    
    # if no backends available, default to triton
    if not available:
        raise ValueError("No Backends available")
        
    return available

@lru_cache()
def get_fn_params(fn_name):
    # get params for fn
    packing = get_packing_type(fn_name)
    is_varlen = True if "varlen" in fn_name else False
    is_fp8 = True if "fp8" in fn_name else False
    supported_dtypes = SUPPORTED_DTYPES.get(fn_name, [torch.float16])  # default to float16 if not found
    supported_backends = [backend for backend in SUPPORTED_BACKENDS.get(fn_name, ["triton"]) if backend in available_backends()]  # default to triton backend
    supports_backward = False if fn_name in ["flash_attn_with_kvcache"] else True
    supported_modes = SUPPORTED_MODES.get(fn_name, ["fwd"])
    device = "cuda"

    # check backward pass support
    if not supports_backward:
        warning(f"{fn_name} does not have a backward pass so benching forward pass only.")

    return is_varlen, is_fp8, packing, supported_dtypes, supported_backends, supported_modes, device

def generate_fn_inputs(
    fn_name: str,
    BATCH: int,
    HQ: int,
    HK: int,
    N_CTX_Q: int,
    N_CTX_K: int,
    D_HEAD: int,
    CAUSAL: bool,
    DROPOUT_P: float,
    dtype: torch.dtype,
    device: Literal["cpu", "cuda"]
    ):
    if fn_name == "flash_attn_func":
        return input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, CAUSAL, DROPOUT_P, dtype, layout="bshd", device=device)
    elif fn_name == "flash_attn_kvpacked_func":
        return input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, CAUSAL, DROPOUT_P, dtype, layout="bshd", packing="kv", device=device)
    elif fn_name == "flash_attn_qkvpacked_func":
        return input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, CAUSAL, DROPOUT_P, dtype, layout="bshd", packing="qkv", device=device)
    elif fn_name == "flash_attn_varlen_func":
        return input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, CAUSAL, DROPOUT_P, dtype, layout="thd", device=device) 
    elif fn_name == "flash_attn_varlen_kvpacked_func":
        return input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, CAUSAL, DROPOUT_P, dtype, layout="thd", packing="kv", device=device)
    elif fn_name == "flash_attn_varlen_qkvpacked_func":
        return input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, CAUSAL, DROPOUT_P, dtype, layout="thd", packing="qkv", device=device)
    elif fn_name == "flash_attn_with_kvcache":
        return input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, CAUSAL, DROPOUT_P, dtype, layout="bshd", device=device)
    elif fn_name == "flash_attn_fp8_func":
        return input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, CAUSAL, DROPOUT_P, dtype, layout="bshd", device=device)
    elif fn_name == "flash_attn_qkvpacked_fp8_func":
        return input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, CAUSAL, DROPOUT_P, dtype, layout="bshd", packing="qkv", device=device)
    elif fn_name == "flash_attn_varlen_fp8_func":
        return input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, CAUSAL, DROPOUT_P, dtype, layout="thd", device=device)
    elif fn_name == "flash_attn_varlen_qkvpacked_fp8_func":
        return input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, CAUSAL, DROPOUT_P, dtype, layout="thd", packing="qkv", device=device)
    else:
        valid_fn_names = ", ".join(FUNCTIONS)
        raise ValueError(f"{fn_name} should be one of the following functions. {valid_fn_names}")

def estimate_memory(config):
    batch, hq, hk, sq, sk, d_head, causal, dropout = config
    memory_estimate = batch * (hq * sq + hk * sk) * d_head * 4  # bytes
    return memory_estimate

def generate_benchmark_configs(is_varlen: bool, packing: Optional[Literal["kv", "qkv"]]):
    """
    generates a small number of configs that cover the parameter space well
    """
    
    # define all parameter options as lists
    batch_sizes = [1, 64]
    if packing == "qkv":
        hq_values = hk_values = [2, 8]
        sq_values = sk_values = [256, 8192]
    else:
        if is_varlen: # make sure the seqlen is greater than the batchsize so that subsequences are greater than 0
            hq_values = [16, 32] # test mqa/gqa
            hk_values = [8, 16]
            sq_values = [128, 512]
            sk_values = [512, 2024]
        else:
            hq_values = [64, 128] # test mqa/gqa
            hk_values = [16, 64]
            sq_values = [4, 4096]
            sk_values = [4096, 16384] # test large k values for inference perf
    d_head_values = [64, 128]
    causal_values = [True] # most models usual causal True
    dropout_values = [0.0]
    
    # generate all fn_configs without inputs
    input_configs = []
    
    # one big loop to generate configs
    for batch in batch_sizes:
        for hq in hq_values:
            for hk in hk_values:
                for sq in sq_values:
                    for sk in sk_values:
                        for d_head in d_head_values:
                            for causal in causal_values:
                                for dropout in dropout_values:
                                    # filter configs
                                    input_config = (batch, hq, hk, sq, sk, d_head, causal, dropout)

                                    # skip if memory usage would be too high
                                    if estimate_memory(input_config) > 8 * 1024 * 1024 * 1024:  # 8 GB limit
                                        continue

                                    # we need hq to be a multiple of hk
                                    if hq % hk != 0:
                                        continue

                                    # for qkvpacked functions, q and k must have same dimensions
                                    if packing == "qkv" and (sq != sk or hq != hk):
                                        continue
                                    
                                    input_configs.append(input_config)
    
    return input_configs

def create_benchmark_fn(
    flash_attn,
    fn_name,
    fn_input,
    mode: Literal["fwd", "bwd", "full"]
):
    if DEBUG:
        print("create_benchmark_fn")
        print("flash_attn:", flash_attn)
        print("fn_name:", fn_name)
        print("fn_input:", len(fn_input))
        print("mode:", mode)

    if fn_name == "flash_attn_func":
        q, k, v, do, metadata = fn_input
        if mode == "fwd":
            def flash_attn_bench_fn():
                out, lse, S_dmask = flash_attn.flash_attn_func(
                    q,
                    k,
                    v,
                    metadata.dropout_p,
                    causal=metadata.causal,
                    window_size=(-1, -1),
                    softcap=0.0,
                    alibi_slopes=None,
                    deterministic=False,
                    return_attn_probs=True,
                )
                return out
        elif mode == "bwd":
            out, lse, S_dmask = flash_attn.flash_attn_func(
                q,
                k,
                v,
                metadata.dropout_p,
                causal=metadata.causal,
                window_size=(-1, -1),
                softcap=0.0,
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=True,
            )
            def flash_attn_bench_fn():
                dq, dk, dv = torch.autograd.grad(out, (q, k, v), do, retain_graph=True)
                return dq, dk, dv
        elif mode == "full":
            def flash_attn_bench_fn():
                out, lse, S_dmask = flash_attn.flash_attn_func(
                    q,
                    k,
                    v,
                    metadata.dropout_p,
                    causal=metadata.causal,
                    window_size=(-1, -1),
                    softcap=0.0,
                    alibi_slopes=None,
                    deterministic=False,
                    return_attn_probs=True,
                )
                dq, dk, dv = torch.autograd.grad(out, (q, k, v), do, retain_graph=True)
                return dq, dk, dv
        else:
            raise ValueError(f"Unsupported benchmarking mode: {mode}")

        return flash_attn_bench_fn

    elif fn_name == "flash_attn_kvpacked_func":
        q, kv, do, metadata = fn_input
        if mode == "fwd":
            def flash_attn_kvpacked_bench_fn():
                out, lse, S_dmask = flash_attn.flash_attn_kvpacked_func(
                    q,
                    kv,
                    metadata.dropout_p,
                    causal=metadata.causal,
                    window_size=(-1, -1),
                    softcap=0.0,
                    alibi_slopes=None,
                    deterministic=False,
                    return_attn_probs=True,
                )
                return out
        elif mode == "bwd":
            out, lse, S_dmask = flash_attn.flash_attn_kvpacked_func(
                q,
                kv,
                metadata.dropout_p,
                causal=metadata.causal,
                window_size=(-1, -1),
                softcap=0.0,
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=True,
            )
            def flash_attn_kvpacked_bench_fn():
                dq, dkv = torch.autograd.grad(out, (q, kv), do, retain_graph=True)
                return dq, dkv
        elif mode == "full":
            def flash_attn_kvpacked_bench_fn():
                out, lse, S_dmask = flash_attn.flash_attn_kvpacked_func(
                    q,
                    kv,
                    metadata.dropout_p,
                    causal=metadata.causal,
                    window_size=(-1, -1),
                    softcap=0.0,
                    alibi_slopes=None,
                    deterministic=False,
                    return_attn_probs=True,
                )
                dq, dkv = torch.autograd.grad(out, (q, kv), do, retain_graph=True)
                return dq, dkv
        else:
            raise ValueError(f"Unsupported benchmarking mode: {mode}")

        return flash_attn_kvpacked_bench_fn
    elif fn_name == "flash_attn_qkvpacked_func":
        qkv, do, metadata = fn_input
        if mode == "fwd":
            def flash_attn_qkvpacked_bench_fn():
                out, lse, S_dmask = flash_attn.flash_attn_qkvpacked_func(
                    qkv,
                    metadata.dropout_p,
                    causal=metadata.causal,
                    window_size=(-1, -1),
                    softcap=0.0,
                    alibi_slopes=None,
                    deterministic=False,
                    return_attn_probs=True,
                )
                return out
        elif mode == "bwd":
            out, lse, S_dmask = flash_attn.flash_attn_qkvpacked_func(
                    qkv,
                    metadata.dropout_p,
                    causal=metadata.causal,
                    window_size=(-1, -1),
                    softcap=0.0,
                    alibi_slopes=None,
                    deterministic=False,
                    return_attn_probs=True,
                )
            def flash_attn_qkvpacked_bench_fn():
                dqkv = torch.autograd.grad(out, (qkv), do, retain_graph=True)
                return dqkv
        elif mode == "full":
            def flash_attn_qkvpacked_bench_fn():
                out, lse, S_dmask = flash_attn.flash_attn_qkvpacked_func(
                    qkv,
                    metadata.dropout_p,
                    causal=metadata.causal,
                    window_size=(-1, -1),
                    softcap=0.0,
                    alibi_slopes=None,
                    deterministic=False,
                    return_attn_probs=True,
                )
                dqkv = torch.autograd.grad(out, (qkv), do, retain_graph=True)
                return dqkv
        else:
            raise ValueError(f"Unsupported benchmarking mode: {mode}")

        return flash_attn_qkvpacked_bench_fn
    elif fn_name == "flash_attn_varlen_func":
        q_unpad, k_unpad, v_unpad, do_unpad, metadata = fn_input
        if mode == "fwd":
            def flash_attn_varlen_bench_fn():
                out_unpad, lse, S_dmask = flash_attn.flash_attn_varlen_func(
                    q_unpad,
                    k_unpad,
                    v_unpad,
                    metadata.cu_seqlens_q,
                    metadata.cu_seqlens_k,
                    metadata.max_seqlens_q,
                    metadata.max_seqlens_k,
                    metadata.dropout_p,
                    causal=metadata.causal,
                    window_size=(-1, -1),
                    softcap=0.0 ,
                    alibi_slopes=None,
                    deterministic=False,
                    return_attn_probs=True,
                )
                return out_unpad
        elif mode == "bwd":
            out_unpad, lse, S_dmask = flash_attn.flash_attn_varlen_func(
                    q_unpad,
                    k_unpad,
                    v_unpad,
                    metadata.cu_seqlens_q,
                    metadata.cu_seqlens_k,
                    metadata.max_seqlens_q,
                    metadata.max_seqlens_k,
                    metadata.dropout_p,
                    causal=metadata.causal,
                    window_size=(-1, -1),
                    softcap=0.0 ,
                    alibi_slopes=None,
                    deterministic=False,
                    return_attn_probs=True,
                )
            def flash_attn_varlen_bench_fn():
                dq_unpad, dk_unpad, dv_unpad = torch.autograd.grad(out_unpad, (q_unpad, k_unpad, v_unpad), do_unpad, retain_graph=True)
                return dq_unpad, dk_unpad, dv_unpad
        elif mode == "full":
            def flash_attn_varlen_bench_fn():
                out_unpad, lse, S_dmask = flash_attn.flash_attn_varlen_func(
                    q_unpad,
                    k_unpad,
                    v_unpad,
                    metadata.cu_seqlens_q,
                    metadata.cu_seqlens_k,
                    metadata.max_seqlens_q,
                    metadata.max_seqlens_k,
                    metadata.dropout_p,
                    causal=metadata.causal,
                    window_size=(-1, -1),
                    softcap=0.0 ,
                    alibi_slopes=None,
                    deterministic=False,
                    return_attn_probs=True,
                )
                dq_unpad, dk_unpad, dv_unpad = torch.autograd.grad(out_unpad, (q_unpad, k_unpad, v_unpad), do_unpad, retain_graph=True)
                return dq_unpad, dk_unpad, dv_unpad
        else:
            raise ValueError(f"Unsupported benchmarking mode: {mode}")

        return flash_attn_varlen_bench_fn
    elif fn_name == "flash_attn_varlen_kvpacked_func":
        q_unpad, kv_unpad, do_unpad, metadata = fn_input
        if mode == "fwd":
            def flash_attn_varlen_kvpacked_bench_fn():
                out_unpad, lse, S_dmask = flash_attn.flash_attn_varlen_kvpacked_func(
                    q_unpad,
                    kv_unpad,
                    metadata.cu_seqlens_q,
                    metadata.cu_seqlens_k,
                    metadata.max_seqlens_q,
                    metadata.max_seqlens_k,
                    metadata.dropout_p,
                    causal=metadata.causal,
                    window_size=(-1, -1),
                    softcap=0.0 ,
                    alibi_slopes=None,
                    deterministic=False,
                    return_attn_probs=True,
                )
                return out_unpad
        elif mode == "bwd":
            out_unpad, lse, S_dmask = flash_attn.flash_attn_varlen_kvpacked_func(
                q_unpad,
                kv_unpad,
                metadata.cu_seqlens_q,
                metadata.cu_seqlens_k,
                metadata.max_seqlens_q,
                metadata.max_seqlens_k,
                metadata.dropout_p,
                causal=metadata.causal,
                window_size=(-1, -1),
                softcap=0.0 ,
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=True,
            )
            def flash_attn_varlen_kvpacked_bench_fn():
                dq_unpad, dkv_unpad = torch.autograd.grad(out_unpad, (q_unpad, kv_unpad), do_unpad, retain_graph=True)
                return dq_unpad, dkv_unpad
        elif mode == "full":
            def flash_attn_varlen_kvpacked_bench_fn():
                out_unpad, lse, S_dmask = flash_attn.flash_attn_varlen_kvpacked_func(
                    q_unpad,
                    kv_unpad,
                    metadata.cu_seqlens_q,
                    metadata.cu_seqlens_k,
                    metadata.max_seqlens_q,
                    metadata.max_seqlens_k,
                    metadata.dropout_p,
                    causal=metadata.causal,
                    window_size=(-1, -1),
                    softcap=0.0 ,
                    alibi_slopes=None,
                    deterministic=False,
                    return_attn_probs=True,
                )
                dq_unpad, dkv_unpad = torch.autograd.grad(out_unpad, (q_unpad, kv_unpad), do_unpad, retain_graph=True)
                return dq_unpad, dkv_unpad
        else:
            raise ValueError(f"Unsupported benchmarking mode: {mode}")

        return flash_attn_varlen_kvpacked_bench_fn
    elif fn_name == "flash_attn_varlen_qkvpacked_func":
        qkv_unpad, do_unpad, metadata = fn_input
        if mode == "fwd":
            def flash_attn_varlen_qkvpacked_bench_fn():
                out_unpad, lse, S_dmask = flash_attn.flash_attn_varlen_qkvpacked_func(
                    qkv_unpad,
                    metadata.cu_seqlens_q,
                    metadata.max_seqlens_q,
                    metadata.dropout_p,
                    causal=metadata.causal,
                    window_size=(-1, -1),
                    softcap=0.0 ,
                    alibi_slopes=None,
                    deterministic=False,
                    return_attn_probs=True,
                )
                return out_unpad
        elif mode == "bwd":
            out_unpad, lse, S_dmask = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv_unpad,
                metadata.cu_seqlens_q,
                metadata.max_seqlens_q,
                metadata.dropout_p,
                causal=metadata.causal,
                window_size=(-1, -1),
                softcap=0.0 ,
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=True,
            )
            def flash_attn_varlen_qkvpacked_bench_fn():
                dqkv_unpad = torch.autograd.grad(out_unpad, (qkv_unpad), do_unpad, retain_graph=True)
                return dqkv_unpad
        elif mode == "full":
            def flash_attn_varlen_qkvpacked_bench_fn():
                out_unpad, lse, S_dmask = flash_attn.flash_attn_varlen_qkvpacked_func(
                    qkv_unpad,
                    metadata.cu_seqlens_q,
                    metadata.max_seqlens_q,
                    metadata.dropout_p,
                    causal=metadata.causal,
                    window_size=(-1, -1),
                    softcap=0.0 ,
                    alibi_slopes=None,
                    deterministic=False,
                    return_attn_probs=True,
                )
                dqkv_unpad = torch.autograd.grad(out_unpad, (qkv_unpad), do_unpad, retain_graph=True)
                return dqkv_unpad
        else:
            raise ValueError(f"Unsupported benchmarking mode: {mode}")

        return flash_attn_varlen_qkvpacked_bench_fn
    elif fn_name == "flash_attn_with_kvcache":
        q, k_cache, v_cache, _, metadata = fn_input
        if mode == "fwd":
            def flash_attn_with_kvcache_bench_fn():
                out = flash_attn.flash_attn_with_kvcache(
                    q,
                    k_cache,
                    v_cache,
                    None,
                    None,
                    rotary_cos=None,
                    rotary_sin=None,
                    cache_seqlens=None,
                    cache_batch_idx=None,
                    cache_leftpad=None,
                    block_table=None,
                    causal=metadata.causal,
                    window_size=(-1, -1),
                    rotary_interleaved=False,
                    alibi_slopes=None,
                    num_splits=0,
                )
                return out
        else:
            raise ValueError(f"Unsupported benchmarking mode: {mode}")

        return flash_attn_with_kvcache_bench_fn
    elif fn_name == "flash_attn_fp8_func":
        (q, descale_q), (k, descale_k), (v, descale_v), (do, descale_do), metadata = fn_input
        if mode == "fwd":
            def flash_attn_f8_bench_fn():
                out, lse, S_dmask = flash_attn.flash_attn_fp8_func(
                    q,
                    k,
                    v,
                    metadata.dropout_p,
                    causal=metadata.causal,
                    window_size=(-1, -1),
                    softcap=0.0,
                    alibi_slopes=None,
                    deterministic=False,
                    return_attn_probs=True,
                )
                return out
        elif mode == "bwd":
            out, lse, S_dmask = flash_attn.flash_attn_fp8_func(
                q,
                k,
                v,
                metadata.dropout_p,
                causal=metadata.causal,
                window_size=(-1, -1),
                softcap=0.0,
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=True,
            )
            def flash_attn_f8_bench_fn():
                dq, dk, dv = torch.autograd.grad(out, (q, k, v), do, retain_graph=True)
                return dq, dk, dv
        elif mode == "full":
            def flash_attn_f8_bench_fn():
                out, lse, S_dmask = flash_attn.flash_attn_fp8_func(
                    q,
                    k,
                    v,
                    metadata.dropout_p,
                    causal=metadata.causal,
                    window_size=(-1, -1),
                    softcap=0.0,
                    alibi_slopes=None,
                    deterministic=False,
                    return_attn_probs=True,
                )
                dq, dk, dv = torch.autograd.grad(out, (q, k, v), do, retain_graph=True)
                return dq, dk, dv
        else:
            raise ValueError(f"Unsupported benchmarking mode: {mode}")

        return flash_attn_f8_bench_fn
    elif fn_name == "flash_attn_qkvpacked_fp8_func":
        qkv, do, metadata = fn_input
        if mode == "fwd":
            def flash_attn_qkvpacked_fp8_bench_fn():
                out, lse, S_dmask = flash_attn.flash_attn_qkvpacked_fp8_func(
                    qkv,
                    metadata.dropout_p,
                    causal=metadata.causal,
                    window_size=(-1, -1),
                    softcap=0.0,
                    alibi_slopes=None,
                    deterministic=False,
                    return_attn_probs=True,
                )
                return out
        elif mode == "bwd":
            out, lse, S_dmask = flash_attn.flash_attn_qkvpacked_fp8_func(
                    qkv,
                    metadata.dropout_p,
                    causal=metadata.causal,
                    window_size=(-1, -1),
                    softcap=0.0,
                    alibi_slopes=None,
                    deterministic=False,
                    return_attn_probs=True,
                )
            def flash_attn_qkvpacked_fp8_bench_fn():
                dqkv = torch.autograd.grad(out, (qkv), do, retain_graph=True)
                return dqkv
        elif mode == "full":
            def flash_attn_qkvpacked_fp8_bench_fn():
                out, lse, S_dmask = flash_attn.flash_attn_qkvpacked_fp8_func(
                    qkv,
                    metadata.dropout_p,
                    causal=metadata.causal,
                    window_size=(-1, -1),
                    softcap=0.0,
                    alibi_slopes=None,
                    deterministic=False,
                    return_attn_probs=True,
                )
                dqkv = torch.autograd.grad(out, (qkv), do, retain_graph=True)
                return dqkv
        else:
            raise ValueError(f"Unsupported benchmarking mode: {mode}")

        return flash_attn_qkvpacked_fp8_bench_fn
    elif fn_name == "flash_attn_varlen_fp8_func":
        (q_unpad, descale_q), (k_unpad, descale_k), (v_unpad, descale_v), (do_unpad, descale_do), metadata = fn_input
        if mode == "fwd":
            def flash_attn_varlen_fp8_bench_fn():
                out_unpad, lse, S_dmask = flash_attn.flash_attn_varlen_fp8_func(
                    q_unpad,
                    k_unpad,
                    v_unpad,
                    metadata.cu_seqlens_q,
                    metadata.cu_seqlens_k,
                    metadata.max_seqlens_q,
                    metadata.max_seqlens_k,
                    metadata.dropout_p,
                    causal=metadata.causal,
                    window_size=(-1, -1),
                    softcap=0.0 ,
                    alibi_slopes=None,
                    deterministic=False,
                    return_attn_probs=True,
                )
                return out_unpad
        elif mode == "bwd":
            out_unpad, lse, S_dmask = flash_attn.flash_attn_varlen_fp8_func(
                    q_unpad,
                    k_unpad,
                    v_unpad,
                    metadata.cu_seqlens_q,
                    metadata.cu_seqlens_k,
                    metadata.max_seqlens_q,
                    metadata.max_seqlens_k,
                    metadata.dropout_p,
                    causal=metadata.causal,
                    window_size=(-1, -1),
                    softcap=0.0 ,
                    alibi_slopes=None,
                    deterministic=False,
                    return_attn_probs=True,
                )
            def flash_attn_varlen_fp8_bench_fn():
                dq_unpad, dk_unpad, dv_unpad = torch.autograd.grad(out_unpad, (q_unpad, k_unpad, v_unpad), do_unpad, retain_graph=True)
                return dq_unpad, dk_unpad, dv_unpad
        elif mode == "full":
            def flash_attn_varlen_fp8_bench_fn():
                out_unpad, lse, S_dmask = flash_attn.flash_attn_varlen_fp8_func(
                    q_unpad,
                    k_unpad,
                    v_unpad,
                    metadata.cu_seqlens_q,
                    metadata.cu_seqlens_k,
                    metadata.max_seqlens_q,
                    metadata.max_seqlens_k,
                    metadata.dropout_p,
                    causal=metadata.causal,
                    window_size=(-1, -1),
                    softcap=0.0 ,
                    alibi_slopes=None,
                    deterministic=False,
                    return_attn_probs=True,
                )
                dq_unpad, dk_unpad, dv_unpad = torch.autograd.grad(out_unpad, (q_unpad, k_unpad, v_unpad), do_unpad, retain_graph=True)
                return dq_unpad, dk_unpad, dv_unpad
        else:
            raise ValueError(f"Unsupported benchmarking mode: {mode}")

        return flash_attn_varlen_fp8_bench_fn
    elif fn_name == "flash_attn_varlen_qkvpacked_fp8_func":
        qkv_unpad, do_unpad, metadata = fn_input
        if mode == "fwd":
            def flash_attn_varlen_qkvpacked_fp8_bench_fn():
                out_unpad, lse, S_dmask = flash_attn.flash_attn_varlen_qkvpacked_fp8_func(
                    qkv_unpad,
                    metadata.cu_seqlens_q,
                    metadata.max_seqlens_q,
                    metadata.dropout_p,
                    causal=metadata.causal,
                    window_size=(-1, -1),
                    softcap=0.0 ,
                    alibi_slopes=None,
                    deterministic=False,
                    return_attn_probs=True,
                )
                return out_unpad
        elif mode == "bwd":
            out_unpad, lse, S_dmask = flash_attn.flash_attn_varlen_qkvpacked_fp8_func(
                qkv_unpad,
                metadata.cu_seqlens_q,
                metadata.max_seqlens_q,
                metadata.dropout_p,
                causal=metadata.causal,
                window_size=(-1, -1),
                softcap=0.0 ,
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=True,
            )
            def flash_attn_varlen_qkvpacked_fp8_bench_fn():
                dqkv_unpad = torch.autograd.grad(out_unpad, (qkv_unpad), do_unpad, retain_graph=True)
                return dqkv_unpad
        elif mode == "full":
            def flash_attn_varlen_qkvpacked_fp8_bench_fn():
                out_unpad, lse, S_dmask = flash_attn.flash_attn_varlen_qkvpacked_fp8_func(
                    qkv_unpad,
                    metadata.cu_seqlens_q,
                    metadata.max_seqlens_q,
                    metadata.dropout_p,
                    causal=metadata.causal,
                    window_size=(-1, -1),
                    softcap=0.0 ,
                    alibi_slopes=None,
                    deterministic=False,
                    return_attn_probs=True,
                )
                dqkv_unpad = torch.autograd.grad(out_unpad, (qkv_unpad), do_unpad, retain_graph=True)
                return dqkv_unpad
        else:
            raise ValueError(f"Unsupported benchmarking mode: {mode}")

        return flash_attn_varlen_qkvpacked_fp8_bench_fn
    else:
        valid_fn_names = ", ".join(FUNCTIONS)
        raise ValueError(f"{fn_name} should be one of the following functions. {valid_fn_names}")

def get_packing_type(fn_name: str) -> Optional[Literal["kv", "qkv"]]:
    if "_kvpacked" in fn_name:
        packing = "kv"
    elif "_qkvpacked" in fn_name:
        packing = "qkv"
    else:
        packing = None

    return packing

def load_flash_attn_module(backend: Literal["triton", "ck"], verbose = False):
    """
    Load the flash_attn module with the specified backend configuration
    """

    # remove any existing env variables first
    for key in ENV_FLAGS:
        if key in os.environ:
            del os.environ[key]

    # set environment variable for the desired backend
    if backend == "triton":
        os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"
        os.environ["FLASH_ATTENTION_TRITON_AMD_AUTOTUNE"] = "1"
        os.environ["FLASH_ATTENTION_TRITON_AMD_DEBUG"] = "0"
    elif backend == "ck":
        os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "FALSE"
    else:
        raise ValueError(f"Unknown backend {backend}")
    
    if verbose:
        print(f"Loading flash_attn module with {backend} backend.")
    
    # Remove any existing flash_attn modules from sys.modules
    for module_name in list(sys.modules.keys()):
        if module_name.startswith('flash_attn'):
            del sys.modules[module_name]
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Import and return the module
    import flash_attn
    
    return flash_attn

def run_benchmark(func_config, input_configs):
    """
    Runs the benchmark for the provided function configuration with the given input configurations.
    """
    # print new line to seperate benchmark runs
    print()
    if DEBUG:
        print("func_config:", func_config)

    # extract function configuration parameters
    fn_name = func_config.fn_name
    mode = func_config.mode
    dtype = func_config.dtype
    backend = func_config.backend

    # load flash attention module
    flash_attn_module = load_flash_attn_module(backend, verbose=True)
 
    # start timing the benchmark
    start_time = time.time()

    # print bench fn
    mode_text_dict = {
        "fwd": "forward",
        "bwd": "backward",
        "full": "forward and backward"
    }
    print(f"Benchmarking {fn_name} {mode_text_dict[mode]} with {len(input_configs)} configs in {dtype} on the {backend} backend ...")

    # Setup benchmark configurations
    bench_configs = [
        triton.testing.Benchmark(
            x_names=["BATCH", "HQ", "HK", "N_CTX_Q", "N_CTX_K", "D_HEAD", "CAUSAL", "DROPOUT"],
            x_vals=list(input_configs.keys()),
            line_arg="provider",
            line_vals=["triton"],
            line_names=["Time (ms)"],
            styles=[("red", "-")],
            ylabel="ms",
            plot_name=f"benchmark-{func_config}",
            args={
            },
        )
    ]

    @triton.testing.perf_report(bench_configs)
    def bench_function(
        BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, CAUSAL, DROPOUT, provider, device="cuda"
    ):
        if DEBUG:
            print("BATCH:", BATCH)
            print("HQ:", HQ)
            print("HK:", HK)
            print("N_CTX_Q:", N_CTX_Q)
            print("N_CTX_Q:", N_CTX_Q)
            print("D_HEAD:", D_HEAD)
            print("CAUSAL:", CAUSAL)
            print("DROPOUT:", DROPOUT)
            print("mode:", mode)
            print("provider:", provider)
            print("device:", device)
        fn_input = input_configs[(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, CAUSAL, DROPOUT)]
        benchmark_fn = create_benchmark_fn(flash_attn_module, fn_name, fn_input, mode)

        # run the benchmark
        ms = triton.testing.do_bench(benchmark_fn, warmup=25, rep=100)
        return ms

    df = bench_function.run(save_path=".", print_data=True, return_df=True)[0]
    
    # set the column name to reflect the function configuration
    df = df.rename(columns={"Time (ms)": func_config.column_name()})
    
    # calculate and print elapsed time
    elapsed_time = time.time() - start_time
    print(f"Total time for benchmarking {fn_name} in {mode} mode with {dtype}: {elapsed_time:.2f} seconds")

    return df

class FunctionConfig:
    def __init__(self, fn_name: str, mode: Literal["fwd", "bwd", "full"], dtype, backend: Literal["triton", "ck"]):
        self.fn_name = fn_name
        self.mode: Literal["fwd", "bwd", "full"] = mode
        self.dtype = dtype
        self.backend: Literal["triton", "ck"] = backend
        self.arch = get_arch()
    
    def __str__(self):
        # extract base dtype name if it's a torch dtype
        dtype_str = str(self.dtype)
        if "torch." in dtype_str:
            dtype_str = dtype_str.split(".")[-1]
        return f"{self.fn_name}_{self.mode}_{dtype_str}_{self.backend}_{self.arch}"
    
    def column_name(self):
        return f"{self}_ms"

def process_args():
    """
    Parses command-line arguments and returns function configs and input configs.
    """
    # create parser
    parser = argparse.ArgumentParser(
        prog="Benchmark FlashAttention",
        allow_abbrev=False,
    )
    # functions
    parser.add_argument(
        "-benchmark_fn",
        type=str,
        nargs="*",
        choices=FUNCTIONS,
        help=f"Function(s) to benchmark",
    )
    # config
    parser.add_argument("-b", type=int, default=None, help="Batch size")
    parser.add_argument("-hq", type=int, default=None, help="Q Number of heads")
    parser.add_argument("-hk", type=int, default=None, help="K and V Number of heads")
    parser.add_argument("-sq", type=int, default=None, help="Q Sequence Length")
    parser.add_argument("-sk", type=int, default=None, help="K and V Sequence Length")
    parser.add_argument("-d", type=int, default=None, help="Head Dimension")
    parser.add_argument("-causal", action="store_true", default=None, help="Causal")
    parser.add_argument("-dropout", type=float, default=None, help="Dropout")

    # parse args
    args = parser.parse_args()

    # determine the functions to benchmark
    if args.benchmark_fn is None or len(args.benchmark_fn) == 0:
        raise ValueError(f"Provide a function to benchmark from this list: {FUNCTIONS}")
    else:
        for fn_name in args.benchmark_fn:
            if fn_name not in FUNCTIONS:
                raise ValueError(f"invalid benchmark function specified: {fn_name}")
        benchmark_fns = args.benchmark_fn

    # fenerate function configurations and input configurations separately
    function_configs = []
    all_input_configs = {}  # Maps function config -> input configs
    
    for fn_name in benchmark_fns:
        is_varlen, is_fp8, packing, supported_dtypes, supported_backends, supported_modes, device = get_fn_params(fn_name)
        
        # Generate or use custom input configurations
        if args.b or args.hq or args.hk or args.sq or args.sk or args.d:
            assert args.b and args.hq and args.sq and args.d, (
                "if custom config is specified, please provide at least batch, number of Q heads, Q sequence length, and head size."
            )
            
            batch = args.b
            hq = args.hq
            hk = args.hk if args.hk is not None else args.hq
            sq = args.sq
            sk = args.sk if args.sk is not None else args.sq
            d_head = args.d
            causal = args.causal if args.causal is not None else False
            dropout = args.dropout if args.dropout is not None else 0.0
            input_configs = [(batch, hq, hk, sq, sk, d_head, causal, dropout)]
        else:
            config_type = "llama"
            if config_type == "llama":
                # batch, hq, hk, sq, sk, d_head, causal, dropout
                input_configs = [
                    # LLaMA 3 8B
                    (1, 32, 8, 8192, 8192, 128, True, 0.0),
                    # LLaMA 3 70B
                    (1, 64, 8, 8192, 8192, 128, True, 0.0),
                    # # LLaMA 3.1 8B
                    # (1, 32, 8, 128000, 128000, 128, True, 0.1),
                    # # LLaMA 3.1 70B
                    # (1, 64, 8, 128000, 128000, 128, True, 0.1),
                    # # LLaMA 3.1 405B
                    # (1, 128, 16, 128000, 128000, 128, True, 0.1),
                ]
            else:
                input_configs = generate_benchmark_configs(is_varlen, packing)
        
        # Create a function config for each backend and dtype combination
        for backend in supported_backends:
            for dtype in supported_dtypes:
                for mode in supported_modes:
                    func_config = FunctionConfig(fn_name, mode, dtype, backend)
                    function_configs.append(func_config)
                    
                    # Generate inputs for this function configuration
                    fn_inputs = {}
                    for input_config in input_configs:
                        fn_inputs[input_config] = generate_fn_inputs(fn_name, *input_config, dtype, device)
                    
                    all_input_configs[func_config] = fn_inputs

    return function_configs, all_input_configs

def check_environment_variables():
    for key in ENV_FLAGS:
        if key in os.environ:
            raise ValueError(f"Running with {key} environment variable is not recommended for the benching script. Use --help to see how to use the benching script.")

def main():
    """
    Main function to run benchmarks.
    """
    # check environment variables
    check_environment_variables()

    # start timing the entire benchmarking process
    total_start_time = time.time()

    # process args to get function configs and input configs
    function_configs, all_input_configs = process_args()
    
    # Check if we have multiple function configurations
    has_multiple_func_configs = len(function_configs) > 1
    combined_df = None

    # run benchmarks for each function configuration
    for func_config in function_configs:
        # run benchmark with the input configs for this function config
        input_configs = all_input_configs[func_config]
        df = run_benchmark(func_config, input_configs)
        
        # Define the columns that represent input configurations
        input_config_cols = ["BATCH", "HQ", "HK", "N_CTX_Q", "N_CTX_K", "D_HEAD", "CAUSAL", "DROPOUT"]
        
        # merge into one final dataframe
        if combined_df is None:
            combined_df = df
        else:
            # Ensure we're joining on input configuration columns
            combined_df = combined_df.merge(df, on=input_config_cols, how="outer")
    

    # print new line to seperate the combined data information from the benchmark specific information
    print()

    # print total time for all benchmarks
    total_elapsed_time = time.time() - total_start_time
    print(f"Total time for all benchmarks: {total_elapsed_time:.2f} seconds")

    # save combined data and make comparisons if we have multiple function configs
    if has_multiple_func_configs:
        if len(function_configs) == 2:
            func1 = function_configs[0]
            func2 = function_configs[1]
            
            # construct column names for the timing results
            col1 = func1.column_name()
            col2 = func2.column_name()
            
            # Check if we're comparing triton vs ck (in either order)
            is_triton_vs_ck = (
                (func1.backend == "triton" and func2.backend == "ck") or
                (func1.backend == "ck" and func2.backend == "triton")
            )
            
            # For triton vs ck comparisons
            if is_triton_vs_ck:
                # For triton vs ck comparisons, always make triton the baseline
                if func1.backend == "triton" and func2.backend == "ck":
                    triton_col = col1
                    ck_col = col2
                    ratio_col = f"ck_to_triton_ratio"
                else:
                    triton_col = col2
                    ck_col = col1
                    ratio_col = f"ck_to_triton_ratio"
                    
                # Calculate ratio: ck_time / triton_time (values > 1 mean triton is faster)
                combined_df[ratio_col] = combined_df[ck_col] / combined_df[triton_col]
                
                # print explanation
                print(f"Comparison Results (triton vs ck):")
                print(f"Ratio values: values > 1 mean triton is faster (by that factor), values < 1 mean ck is faster")
            else:
                # For other comparisons, use the standard approach
                ratio_col = f"{func1}_to_{func2}_ratio"
                
                # Calculate the ratio
                combined_df[ratio_col] = combined_df[col2] / combined_df[col1]
                
                # print explanation
                print(f"Comparison Results ({func1} vs {func2}):")
                print(f"Ratio values: values > 1 mean {func1} is faster than {func2} (by that factor), values < 1 mean slower")
       
        print(f"Combined data:")
        print(combined_df)

        # save csv & markdown
        combined_filename = f"benchmark_combined"
        combined_df.to_csv(f"{combined_filename}.csv", index=False)
        with open(f"{combined_filename}.md", 'w') as f:
            f.write(combined_df.to_markdown(index=False, floatfmt=".2f"))

if __name__ == "__main__":
    main()