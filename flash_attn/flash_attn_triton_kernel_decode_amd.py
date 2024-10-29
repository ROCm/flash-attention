import math
from typing import Optional, Union
from einops import rearrange, repeat
from flash_attn.layers.rotary import apply_rotary_emb
import pytest
import torch
import sys

import pdb

import triton
import triton.language as tl
from flash_attn.flash_attn_triton_kernel_prefill_amd import MetaData

def _strides(x: torch.Tensor, *stride_names: str):
    if x is None:
        return {f"stride_{s}": 0 for i, s in enumerate(stride_names)}

    assert x.ndim == len(stride_names)
    return {f"stride_{s}": x.stride(i) for i, s in enumerate(stride_names)}

@triton.jit
def rotary_kernel_splitk(
    # Dimensions of X
    X,              # tensor being rotated. Has shape (batch (z), seqlen (s), group (g), head (h), head_dim (d))
    seqlen_x,       # seqlen of the x dim. shape is (batch (z), )
    head_dim,
    rotary_dim,     # size of embedding space we end up rotating

    # COS/SIN and Offsetting Into It
    COS,            # tensor of shape (seqlen (m), ro_dim // 2)
    SIN,            # tensor of shape (seqlen (m), ro_dim // 2)
    SEQLEN_OFFSET,  # we use this as an offset into COS and SIN to apply the correct rotation
    SEQLEN_OFFSET_IS_TENSOR: tl.constexpr, # if seqlen_offset is a tensor it has shape (num_batch, )
    
    # PID Offsets
    batch_pid: tl.constexpr,      # pid for batch
    start_m: tl.constexpr,        # the token idx the current M_BLOCK starts at.
    group_pid: tl.constexpr,      # pid for group
    head_pid: tl.constexpr,       # pid to access head

    # Strides
    stride_batch: tl.constexpr,
    stride_m: tl.constexpr,
    stride_group: tl.constexpr,
    stride_head: tl.constexpr,
    stride_headdim: tl.constexpr,

    # Misc
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
    TRANSPOSE: tl.constexpr,

    # Meta-parameters
    BLOCK_M: tl.constexpr,     # block size to access chunks of tokens (# of tokens simultaneously)
    BLOCK_K: tl.constexpr,     # block size to access chunks of headdim (# of dimensions processed)
):
    """
    Note: 
    - for K in splitk let BLOCK_M = BLOCK_N, and start_m=start_n
    """
    # pdb.set_trace()
    range_m = start_m + tl.arange(0, BLOCK_M)
    range_d = tl.arange(0, BLOCK_K)

    x_ptr = X + (batch_pid * stride_batch) + (group_pid * stride_group) + (head_pid * stride_head)   # pointer to x block
    x_mask = (range_m < seqlen_x)[:, None] & (range_d < rotary_dim)[None, :]

    ro_dim_half = rotary_dim // 2       # length of cos/sin

    if SEQLEN_OFFSET_IS_TENSOR:
        seqlen_offset = tl.load(SEQLEN_OFFSET + batch_pid) # a tensor
    else:
        seqlen_offset = SEQLEN_OFFSET # an int

    # load full x (puts values in cache)
    x_range = range_m[:, None]*stride_m + range_d[None, :]
    x_mask = (range_m < seqlen_x)[:, None] & (range_d < head_dim)[None, :]
    x = tl.load(x_ptr + x_range, mask=x_mask)


    if not INTERLEAVED:
        range_d_half_duplicate = range_d % (rotary_dim // 2)

        x0_range = range_m[:, None]*stride_m + range_d_half_duplicate[None, :]*stride_headdim                # BLOCK_M x 1st half of headdim (fast to load)
        x1_range = range_m[:, None]*stride_m + range_d_half_duplicate[None, :]*stride_headdim + ro_dim_half  # BLOCK_M x 2nd half of headdim (fast to load)

        x0_mask = (range_m < seqlen_x)[:, None] & (range_d_half_duplicate < rotary_dim)[None, :]                  # Mask for the first half
        x1_mask = (range_m < seqlen_x)[:, None] & (range_d_half_duplicate + ro_dim_half < rotary_dim)[None, :]    # Mask for the second half

        range_m_cos_sin = range_m + seqlen_offset # offsets cos and sin based on current m position range and seqlen offset
        COS = COS + (range_m_cos_sin[:, None] * ro_dim_half + range_d_half_duplicate[None, :])
        SIN = SIN + (range_m_cos_sin[:, None] * ro_dim_half + range_d_half_duplicate[None, :])
        cos = tl.load(
            COS, mask=(range_m[:, None] < seqlen_x) & (range_d_half_duplicate[None, :] < ro_dim_half), other=1.0
        ).to(tl.float32)
        sin = tl.load(
            SIN, mask=(range_m[:, None] < seqlen_x + seqlen_offset) & (range_d_half_duplicate[None, :] < ro_dim_half), other=0.0
        ).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        
        x0 = tl.load(x_ptr + x0_range, mask=x0_mask).to(tl.float32)
        x1 = tl.load(x_ptr + x1_range, mask=x1_mask).to(tl.float32)

        # Rotate corresponding elements in each half
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos

        out = tl.where(range_d[None, :] // ro_dim_half == 0, o0, o1)

        # for all dim not in rotary_dim, leave untouched
        out = tl.where(range_d[None, :] < rotary_dim, out, x)

        # transpose the rotated vector 
        if TRANSPOSE:
            out = tl.trans(out)

        return out
        
    else:
        # Interleaved is slow due to x1 load
        range_d_swap = range_d + ((range_d + 1) % 2) * 2 - 1            # 1, 0, 3, 2, 5, 4, ...

        # X Range
        x0_range = range_m[:, None]*stride_m + range_d[None, :]         # 0, 1, 2, 3, 4, 5, ... (fast to load)
        x1_range = range_m[:, None]*stride_m + range_d_swap[None, :]    # 1, 0, 3, 2, 5, 4, ... (slow to load)
        
        # X Masks
        x0_mask = (range_m < seqlen_x)[:, None] & (range_d < rotary_dim)[None, :]                  # Mask for the first half
        x1_mask = (range_m < seqlen_x)[:, None] & (range_d_swap < rotary_dim)[None, :]    # Mask for the second half
        
        # Load COS/SIN
        range_d_repeat = tl.arange(0, BLOCK_K) // 2                # 0, 0, 1, 1, 2, 2, ...

        range_m_cos_sin = range_m + seqlen_offset
        COS = COS + (range_m_cos_sin[:, None] * ro_dim_half + range_d_repeat[None, :])
        SIN = SIN + (range_m_cos_sin[:, None] * ro_dim_half + range_d_repeat[None, :])
        cos = tl.load(
            COS,
            mask=(range_m[:, None] < seqlen_x) & (range_d_repeat[None, :] < ro_dim_half),
            other=1.0,
        ).to(tl.float32)
        sin = tl.load(
            SIN,
            mask=(range_m[:, None] < seqlen_x) & (range_d_repeat[None, :] < ro_dim_half),
            other=0.0,
        ).to(tl.float32)
        if CONJUGATE:
            sin = -sin

        x0 = tl.load(x_ptr + x0_range, mask=x0_mask)
        x1 = tl.load(x_ptr + x1_range, mask=x1_mask)

        x0_cos = x0 * cos
        x1_sin = x1 * sin

        out = tl.where(range_d[None, :] % 2 == 0, x0_cos - x1_sin, x0_cos + x1_sin)

        # for all dim not in rotary_dim, leave untouched
        out = tl.where(range_d[None, :] < rotary_dim, out, x)

        # transpose the rotated vector 
        if TRANSPOSE:
            out = tl.trans(out)

        return out

@triton.jit
def _fwd_kernel_splitK(
    Q,
    K,
    V,
    sm_scale,
    Out_splitK,  # [B, H, split_k, Mq, K]
    Metadata,  # [B, H, 2, split_k, M_ceil] contains [mi, li]
    K_new,
    V_new,
    Cache_seqlens,
    Cache_batch_idx,
    Alibi_slopes,
    # Rotary
    Rotary_cos,
    Rotary_sin,
    Rotary_dim,
    Rotary_interleaved: tl.constexpr,
    Rotary_conjugate: tl.constexpr,
    seqlen_ro,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # Strides
    stride_qz,
    stride_qm,
    stride_qg,
    stride_qh,
    stride_qd,
    stride_kz,
    stride_kn,
    stride_kg,
    stride_kh,
    stride_kd,
    stride_vz,
    stride_vn,
    stride_vg,
    stride_vh,
    stride_vd,
    stride_osk_zhg,
    stride_osk_s,
    stride_osk_m,
    stride_osk_d,
    stride_mzhg,
    stride_m2,
    stride_ms,
    stride_mm,
    stride_kn_z,
    stride_kn_n,
    stride_kn_g,
    stride_kn_h,
    stride_kn_d,
    stride_vn_z,
    stride_vn_n,
    stride_vn_g,
    stride_vn_h,
    stride_vn_d,
    stride_az, 
    stride_ah,
    Z,
    N_CTX_Q,
    N_CTX_K,
    N_CTX_NEW,
    BLOCK_N_PER_SPLIT,
    H_q: tl.constexpr,
    H_kv: tl.constexpr,
    G_q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    ACTUAL_BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BOUNDS_CHECKS_N: tl.constexpr,
    USE_CACHE_SEQLENS: tl.constexpr,
    USE_CACHE_BATCH_IDX: tl.constexpr,
    NEW_KV: tl.constexpr,
    IS_GQA: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    USE_ROTARY: tl.constexpr,
):

    # Padding
    PADDED_HEAD: tl.constexpr = (ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL)
    if PADDED_HEAD:
        d_mask = tl.arange(0, BLOCK_DMODEL) < ACTUAL_BLOCK_DMODEL

    start_m = tl.program_id(0)
    off_zhg = tl.program_id(1)
    off_z = off_zhg // (H_q * G_q)      # batch
    off_h_q = (off_zhg // G_q) % H_q    # head
    off_g_q = off_zhg % G_q             # group (gca / mqa)
    splitk_idx = tl.program_id(2)

    # pick batch index
    if USE_CACHE_BATCH_IDX:
        cache_batch_idx = tl.load(Cache_batch_idx + off_z)
    else:
        cache_batch_idx = off_z

    # Load ALiBi slope if enabled
    if USE_ALIBI:
        a_offset = off_z * stride_az + off_h_q * stride_ah
        alibi_slope = tl.load(Alibi_slopes + a_offset)
    else:
        alibi_slope = None

    lo = splitk_idx * BLOCK_N_PER_SPLIT
    if USE_CACHE_SEQLENS:
        cache_seqlen_last_idx = tl.load(Cache_seqlens + off_z)
        if NEW_KV:
            kv_len = cache_seqlen_last_idx + N_CTX_NEW
        else:
            kv_len = cache_seqlen_last_idx
    else:
        kv_len = N_CTX_K
    hi = tl.minimum((splitk_idx + 1) * BLOCK_N_PER_SPLIT, kv_len)

    HEAD_RATIO: tl.constexpr = H_q // H_kv
    if IS_GQA:
        k_head_idx = off_h_q // HEAD_RATIO
        v_head_idx = k_head_idx
    else:
        k_head_idx = off_h_q
        v_head_idx = off_h_q

    # calculate base offset
    k_base = K + k_head_idx * stride_kh + cache_batch_idx * stride_kz + off_g_q * stride_kg
    v_base = V + v_head_idx * stride_vh + cache_batch_idx * stride_vz + off_g_q * stride_vg

    # Copy new Keys and Values into Cache
    if NEW_KV:
        knew_base = K_new + k_head_idx * stride_kn_h + off_z * stride_kn_z + off_g_q * stride_kn_g
        
        # Determine the starting position for new data in the cache
        if USE_CACHE_SEQLENS:
            start_idx = tl.load(Cache_seqlens + off_z)
        else:
            start_idx = N_CTX_K - N_CTX_NEW

        # Copy new Keys
        for i in range(0, N_CTX_NEW, BLOCK_N):
            # Load from K_new
            k_new_block = tl.load(
                knew_base +
                tl.arange(0, BLOCK_DMODEL)[:, None] * stride_kn_d +
                (tl.arange(0, BLOCK_N) + i)[None, :] * stride_kn_n,
                 mask=(tl.arange(0, BLOCK_N)[None, :] + i < N_CTX_NEW) &
                     (tl.arange(0, BLOCK_DMODEL)[:, None] < ACTUAL_BLOCK_DMODEL),
                other=0
            )

            # pdb.set_trace()

            # apply rotary to k here
            if USE_ROTARY:
                # pdb.set_trace()
                k_new_block = rotary_kernel_splitk(
                    X=K_new,
                    seqlen_x=N_CTX_NEW,
                    head_dim=BLOCK_DMODEL,
                    rotary_dim=Rotary_dim,

                    COS=Rotary_cos,
                    SIN=Rotary_sin,
                    SEQLEN_OFFSET=Cache_seqlens,
                    SEQLEN_OFFSET_IS_TENSOR=IS_SEQLEN_OFFSETS_TENSOR,

                    batch_pid=off_z,
                    start_m=i,              # current block of tokens in new_k
                    group_pid=off_g_q,
                    head_pid=off_h_q,

                    stride_batch= stride_kz,  # batch_strides if not varlen else 0
                    stride_m=stride_kn,
                    stride_group=stride_kg,
                    stride_head=stride_kh,
                    stride_headdim=stride_kd,

                    INTERLEAVED=Rotary_interleaved,
                    CONJUGATE=Rotary_conjugate,
                    TRANSPOSE=True,

                    BLOCK_M=BLOCK_N,
                    BLOCK_K=BLOCK_DMODEL
                )
                # pdb.set_trace()
            
            # Store to K
            tl.store(
                k_base +
                tl.arange(0, BLOCK_DMODEL)[:, None] * stride_kd +
                (tl.arange(0, BLOCK_N) + i + start_idx)[None, :] * stride_kn,
                k_new_block,
                 mask=(tl.arange(0, BLOCK_N)[None, :] + i < N_CTX_NEW) &
                     (tl.arange(0, BLOCK_DMODEL)[:, None] < ACTUAL_BLOCK_DMODEL),
            )

        # Copy new Values
        vnew_base = V_new + v_head_idx * stride_vn_h + off_z * stride_vn_z + off_g_q * stride_vn_g
        for i in range(0, N_CTX_NEW, BLOCK_N):

            # import pdb; pdb.set_trace()
            # Load from V_new
            v_new_block = tl.load(
                vnew_base +
                (tl.arange(0, BLOCK_N) + i)[:, None] * stride_vn_n +
                tl.arange(0, BLOCK_DMODEL)[None, :] * stride_vn_d,
                mask=(tl.arange(0, BLOCK_N)[:, None] + i < N_CTX_NEW) &
                     (tl.arange(0, BLOCK_DMODEL)[None, :] < ACTUAL_BLOCK_DMODEL),
                other=0
            )
            
            # Store to V
            tl.store(
                v_base + 
                (tl.arange(0, BLOCK_N) + i + start_idx)[:, None] * stride_vn +
                tl.arange(0, BLOCK_DMODEL)[None, :] * stride_vd,
                v_new_block,
                 mask=(tl.arange(0, BLOCK_N)[:, None] + i < N_CTX_NEW) &
                     (tl.arange(0, BLOCK_DMODEL)[None, :] < ACTUAL_BLOCK_DMODEL),
            )

    Q_block_ptr = tl.make_block_ptr(
        base=Q + off_h_q * stride_qh + off_z * stride_qz + off_g_q * stride_qg,
        shape=(N_CTX_Q, ACTUAL_BLOCK_DMODEL),
        strides=(stride_qm, stride_qd),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=k_base,
        shape=(ACTUAL_BLOCK_DMODEL, hi),
        strides=(stride_kd, stride_kn),
        offsets=(0, lo),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=v_base,
        shape=(hi, ACTUAL_BLOCK_DMODEL),
        strides=(stride_vn, stride_vd),
        offsets=(lo, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )

    K_scale_shift_block_ptr = None
    V_scale_shift_block_ptr = None

    # initialize pointer to m and l
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)  # noqa: F821

    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    log2_e = 1.44269504
    qk_scale = sm_scale * log2_e

    # import pdb; pdb.set_trace()

    # load q: it will stay in SRAM throughout
    q = tl.load(  # noqa: F821
        tl.advance(Q_block_ptr, (0, 0)),
        boundary_check=(0, )
    )
    q = (q * qk_scale)
    if PADDED_HEAD:
        q = tl.where(d_mask[None, :], q, 0.0)

    # pdb.set_trace()
    if USE_ROTARY:
        q = rotary_kernel_splitk(
                                X=Q,
                                seqlen_x=N_CTX_Q,
                                head_dim=BLOCK_DMODEL,
                                rotary_dim=Rotary_dim,

                                COS=Rotary_cos,
                                SIN=Rotary_sin,
                                SEQLEN_OFFSET=Cache_seqlens,
                                SEQLEN_OFFSET_IS_TENSOR=IS_SEQLEN_OFFSETS_TENSOR,

                                batch_pid=off_z,
                                start_m=start_m*BLOCK_M,
                                group_pid=off_g_q,
                                head_pid=off_h_q,

                                stride_batch= (stride_kz if not IS_VARLEN else 0),  # batch_strides if not varlen else 0
                                stride_m=stride_kn,
                                stride_group=stride_kg,
                                stride_head=stride_kh,
                                stride_headdim=stride_kd,

                                INTERLEAVED=Rotary_interleaved,
                                CONJUGATE=Rotary_conjugate,
                                TRANSPOSE=False,

                                BLOCK_M=BLOCK_M,
                                BLOCK_K=BLOCK_DMODEL
                                )
    # pdb.set_trace()

    
    q = (q * qk_scale)
    if PADDED_HEAD:
        q = tl.where(d_mask[None, :], q, 0.0)

    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        k, v = load_k_v_group(
            K_block_ptr,
            V_block_ptr,
            K_scale_shift_block_ptr,
            V_scale_shift_block_ptr,
            BOUNDS_CHECKS_N,
            1,
            BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL,
            Q.dtype.element_ty,
            0,
        )
        if PADDED_HEAD:
            k = tl.where(d_mask[:, None], k, 0.0)
            v = tl.where(d_mask[None, :], v, 0.0)

        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)

        if USE_ALIBI:
            row_idx = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            col_idx = start_n + tl.arange(0, BLOCK_N)
            
            # Compute relative positions
            relative_pos = row_idx[:, None] + kv_len - (N_CTX_Q + col_idx[None, :])
            relative_pos = tl.abs(relative_pos)
            
            # Compute ALiBi bias
            alibi_bias = -1 * alibi_slope * relative_pos
            qk += (alibi_bias * log2_e)

        # Apply causal mask if IS_CAUSAL is True
        if IS_CAUSAL:
            row_idx = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            col_idx = start_n + tl.arange(0, BLOCK_N)
            
            # create a N_CTX_Q x kv_len causal mask
            col_offset = N_CTX_Q - kv_len
            causal_mask = row_idx[:, None] >= (col_offset + col_idx[None, :])

            # Apply the mask
            qk = tl.where(causal_mask, qk, float("-inf"))

        # TODO: This is slow, and only needed at the last iteration.
        # Maybe we can unroll the last iteration instead?
        if BOUNDS_CHECKS_N:
            qk = tl.where(tl.arange(0, BLOCK_N) < hi - start_n, qk, float("-inf"))

        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        if IS_CAUSAL:
            alpha = tl.math.exp2(tl.where(m_i > float("-inf"), m_i - m_i_new, float("-inf")))
        else:
            alpha = tl.math.exp2(m_i - m_i_new)
        # cause of nan because subtracting infs
        if IS_CAUSAL:
            qk = tl.where(qk > float("-inf"), qk - m_i_new[:, None], float("-inf"))
        else:
            qk = qk - m_i_new[:, None]
        
        p = tl.math.exp2(qk) # p = e^(qk^T)

        # -- update m_i (current max) and l_i (sum of elements) --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        p = p.to(Q.dtype.element_ty)

        # -- scale and update acc --
        acc *= alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # write back O
    O_block_ptr = tl.make_block_ptr(
        base=Out_splitK + off_zhg * stride_osk_zhg + splitk_idx * stride_osk_s,
        shape=(N_CTX_Q, BLOCK_DMODEL),
        strides=(stride_osk_m, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    tl.store(
        tl.advance(O_block_ptr, (0, 0)),
        acc,
        boundary_check=(0, ),
    )
    # Write metadata for split-K reduction
    Metadata_ptr = (Metadata + off_zhg * stride_mzhg + splitk_idx * stride_ms + start_m * BLOCK_M +
                    tl.arange(0, BLOCK_M))
    tl.store(Metadata_ptr, m_i)
    tl.store(Metadata_ptr + stride_m2, l_i)


@triton.jit
def load_k_v_group(
    K_block_ptr,
    V_block_ptr,
    K_scale_shift_block_ptr,
    V_scale_shift_block_ptr,
    BOUNDS_CHECKS_N: tl.constexpr,
    PACKED_PER_VAL: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    ACTUAL_BLOCK_DMODEL: tl.constexpr,
    dtype: tl.constexpr,
    group_id: tl.constexpr,
):
    #Load K/V for a given block

    # Advance to the current quantization group
    K_block_ptr = tl.advance(K_block_ptr, (ACTUAL_BLOCK_DMODEL * group_id, 0))
    V_block_ptr = tl.advance(V_block_ptr, (0, ACTUAL_BLOCK_DMODEL * group_id))

    # -- load k, v --
    k = tl.load(K_block_ptr, boundary_check=(1, ) if BOUNDS_CHECKS_N else ()).to(tl.float32)
    v = tl.load(V_block_ptr, boundary_check=(0, ) if BOUNDS_CHECKS_N else ()).to(tl.float32)

    return k, v


@triton.jit
def cast_uint32_to_half2(scale_shift):
    # Extract two float16 packed into one int32
    scale = scale_shift & 0xFFFF
    shift = scale_shift >> 16
    scale = scale.to(tl.uint16).to(tl.float16, bitcast=True)
    shift = shift.to(tl.uint16).to(tl.float16, bitcast=True)
    return scale, shift


@triton.jit
def dequantize(
    x_,
    scale,
    shift,
    PACKED_PER_VAL: tl.constexpr = 8,
):
    # PACKED_PER_VAL is the number of values packed into
    # each element x_. For example, for int4 quantization
    #and x_ of type int32, PACKED_PER_VAL is 8.

    BLOCK_N: tl.constexpr = x_.shape[0]
    BLOCK_DMODEL_PACKED: tl.constexpr = x_.shape[1]
    offsets = tl.arange(0, PACKED_PER_VAL) * 4
    quant_offset = (x_[:, None, :] >> offsets[None, :, None])  # (BLOCK_N, PACKED_PER_VAL, D // PACKED_PER_VAL)

    quant_offset = tl.view(quant_offset, (BLOCK_N, BLOCK_DMODEL_PACKED * PACKED_PER_VAL))
    # Trick - instead of converting int4 to float16 we view it as float16
    # and then multiply by 32768 * 512 == 2**24
    quant_offset = (quant_offset & 0xF).to(tl.uint16).to(tl.float16, bitcast=True)
    quant_offset = (quant_offset * 32768.0).to(tl.float16)
    scale_512 = scale * 512

    dequant = quant_offset * scale_512 + shift
    return dequant


@triton.jit
def _splitK_reduce(
    Out_splitK,  # [B, H, split_k, Mq, K]
    Metadata,  # [B, H, 2, split_k, M_ceil] contains [mi, li]
    Out,  # [B, H, M, K]
    LSE,  # [B, H, M]
    stride_osk_zhg,
    stride_osk_s,
    stride_osk_m,
    stride_osk_k,
    stride_mzhg,
    stride_m2,
    stride_ms,
    stride_mm,
    stride_oz,
    stride_oh,
    stride_og,
    stride_om,
    stride_ok,
    stride_lse_zhg,
    stride_lse_m,
    M_ceil: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    split_k: tl.constexpr,
    splitK_pow2: tl.constexpr,
    use_mask: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    off_zhg = tl.program_id(0)
    off_z = off_zhg // (H * G)
    off_h = (off_zhg // G) % H
    off_g = off_zhg % G
    off_m = tl.program_id(1)
    off_k = tl.program_id(2)

    # read  chunk
    spk_idx = tl.arange(0, splitK_pow2)
    kidx = tl.arange(0, BLOCK_SIZE)

    Metadata_ptr = (Metadata + stride_mzhg * off_zhg + spk_idx * stride_ms + off_m * stride_mm)

    o_ptr = (Out_splitK + off_zhg * stride_osk_zhg + stride_osk_m * off_m + off_k * BLOCK_SIZE +
             stride_osk_s * spk_idx[:, None] + kidx[None, :] * stride_osk_k)

    # read max values of each splitK
    if use_mask:
        spk_mask = spk_idx < split_k
        l_m = tl.load(Metadata_ptr, mask=spk_mask, other=float("-inf"))
        l_sum = tl.load(Metadata_ptr + stride_m2, mask=spk_mask, other=0.0)
        acc = tl.load(o_ptr, mask=spk_mask[:, None], other=0.0)
    else:
        l_m = tl.load(Metadata_ptr)
        l_sum = tl.load(Metadata_ptr + stride_m2)
        acc = tl.load(o_ptr)

    g_m = tl.max(l_m, axis=0)
    
    if IS_CAUSAL:
        l_m_offset = l_m - g_m
        alpha = tl.where(l_m_offset > float("-inf"), tl.math.exp2(l_m_offset), 0.0)
    else:
        alpha = tl.math.exp2(l_m - g_m)
    # read sum
    l_sum *= alpha
    g_sum = tl.sum(l_sum, axis=0)
    acc = acc * alpha[:, None]

    if IS_CAUSAL:
        # Avoid division by zero
        g_sum_safe = tl.where(g_sum > 0, g_sum, 1.0)
        acc_out = tl.sum(acc, axis=0) / g_sum_safe
    else:
        acc_out = tl.sum(acc, axis=0) / g_sum

    # Store output
    Out_ptr = (Out + stride_oz * off_z + stride_oh * off_h + stride_og * off_g + stride_om * off_m +
               off_k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))
    tl.store(Out_ptr, acc_out)

    # log constant
    log2_e = 1.44269504

    # Store lse
    l_ptrs = LSE + off_zhg * stride_lse_zhg + off_m
    if IS_CAUSAL:
        lse = tl.where(g_sum > 0, (g_m + tl.math.log2(g_sum)) / log2_e, g_m)
        tl.store(l_ptrs, lse)
    else:
        tl.store(l_ptrs, (g_m + tl.math.log2(g_sum)) / log2_e)


def quantize_kv_int4(k: torch.Tensor, num_groups: int = 1) -> torch.Tensor:
    # Scale and shift are such that quantization linearly maps
    # int4 values range [0..15] to input values range min(k)..max(k)
    # individually for every row
    k = k.reshape(*k.shape[:-1], num_groups, k.shape[-1] // num_groups)
    max_vals = torch.max(k, dim=-1, keepdim=True).values
    min_vals = torch.min(k, dim=-1, keepdim=True).values
    scale_k: torch.Tensor = (max_vals - min_vals) / 15

    shift_k = torch.min(k, dim=-1, keepdim=True).values
    scale_k = scale_k.to(torch.float16)
    shift_k = shift_k.to(torch.float16)

    in_bytes = ((k - shift_k.expand(k.shape)) / scale_k.expand(k.shape)) + 0.5
    in_bytes = in_bytes.to(torch.uint8)
    in_int4 = in_bytes & 0xF
    in_int4_packed = in_int4[..., ::2] + (in_int4[..., 1::2] << 4)
    scale_shift = torch.concat([scale_k.view(torch.uint8), shift_k.view(torch.uint8)], dim=-1)
    k_quant = torch.concat(
        [
            scale_shift.flatten(start_dim=-2),
            in_int4_packed.flatten(start_dim=-2),
        ],
        dim=-1,
    ).view(torch.int16)
    return k_quant


def dequantize_kv_fp16(quant_k: torch.Tensor, num_groups: int = 1) -> torch.Tensor:
    k_i16 = quant_k.view(torch.int16)
    k_ui8 = k_i16.view(torch.uint8)

    ss_size = num_groups * 4
    scale_shift_ui8 = k_ui8[..., 0:ss_size]
    scale_shift_ui8 = scale_shift_ui8.reshape(*scale_shift_ui8.shape[:-1], num_groups, 4)
    scale = scale_shift_ui8[..., 0:2].view(torch.float16)
    shift = scale_shift_ui8[..., 2:4].view(torch.float16)

    kv_ui8 = k_ui8[..., ss_size:]
    k_ui8 = kv_ui8.reshape(*kv_ui8.shape[:-1], num_groups, -1)
    k1_i4 = k_ui8 & 0xF
    k2_i4 = (k_ui8 & 0xF0) >> 4
    k_shape = k1_i4.shape
    k1_f16 = k1_i4.to(torch.float16) * scale.expand(k_shape) + shift.expand(k_shape)
    k2_f16 = k2_i4.to(torch.float16) * scale.expand(k_shape) + shift.expand(k_shape)

    out = torch.empty((*k1_f16.shape[:-1], k1_f16.shape[-1] * 2), dtype=torch.float16, device=quant_k.device)
    out[..., ::2] = k1_f16
    out[..., 1::2] = k2_f16
    out = out.reshape(*k_shape[:-2], -1)

    return out


def get_split_k(B: int, G: int, H: int, Mk: int) -> int:
    """Heuristic for the number of splits"""
    bh = max(B * H, 1)  # NOTE: Handle B*h=0 case
    split_k = max(Mk, 1024) // bh
    max_chunk_size = 64
    while split_k > 0 and Mk / split_k < max_chunk_size:
        split_k = split_k // 2
    while B * H * G * split_k >= 1024:
        split_k = split_k // 2
    split_k = min(split_k, 512)
    split_k = max(split_k, 1)
    return split_k

def get_padded_headsize(size):
    # Get closest power of 2 over or equal to 32.
    padded_d_model = 1 << (size - 1).bit_length()
    # Smallest head_dim supported is 16. If smaller, the tile in the
    # kernel is padded - there is no padding in memory for any dims.
    padded_d_model = max(padded_d_model, 16)
    return padded_d_model

class _attention(torch.autograd.Function):

    OPERATOR = _fwd_kernel_splitK
    SUPPORTED_DEVICES = {"cuda"}
    CUDA_MINIMUM_COMPUTE_CAPABILITY = (8, 0)
    SUPPORTED_DTYPES = {
        torch.half,
        torch.bfloat16,
    }
    SUPPORTED_MAX_K = 128
    SUPPORTS_DROPOUT = False
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_BMGHK = True
    NAME = "triton_splitKF"

    @staticmethod
    def forward(cls, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, input_metadata: MetaData):
        original_layout = input_metadata.layout

        # Rotary Embedding Implementation
        # if torch.is_tensor(input_metadata.rotary_cos) and torch.is_tensor(input_metadata.rotary_sin):
        #     if input_metadata.causal or input_metadata.local:
        #         q_ro = apply_rotary_emb(
        #             q,
        #             input_metadata.rotary_cos,
        #             input_metadata.rotary_sin,
        #             seqlen_offsets=input_metadata.cache_seqlens,
        #             interleaved=input_metadata.rotary_interleaved,
        #         )
        #     else:
        #         q_ro = rearrange(
        #             apply_rotary_emb(
        #                 rearrange(q, "b s h d -> b 1 (s h) d"),
        #                 input_metadata.rotary_cos,
        #                 input_metadata.rotary_sin,
        #                 seqlen_offsets=input_metadata.cache_seqlens,
        #                 interleaved=input_metadata.rotary_interleaved,
        #             ),
        #             "b 1 (s h) d -> b s h d",
        #             s=input_metadata.max_seqlens_q,
        #         )
        #     k_ro = apply_rotary_emb(
        #         input_metadata.k_new,
        #         input_metadata.rotary_cos,
        #         input_metadata.rotary_sin,
        #         seqlen_offsets=input_metadata.cache_seqlens,
        #         interleaved=input_metadata.rotary_interleaved,
        #     )

        #     q, input_metadata.k_new = q_ro.to(q.dtype), k_ro.to(q.dtype)

        # kernels expects "bsghd"
        if input_metadata.layout == "bshd":
            q=q.unsqueeze(2)
            k=k.unsqueeze(2)
            v=v.unsqueeze(2)

            if input_metadata.new_kv:
                input_metadata.k_new = input_metadata.k_new.unsqueeze(2)
                input_metadata.v_new = input_metadata.v_new.unsqueeze(2)

            input_metadata.layout = "bsghd"
        elif input_metadata.layout == "bhsd":
            q=q.permute(0, 2, 1, 3).unsqueeze(2)
            k=k.permute(0, 2, 1, 3).unsqueeze(2)
            v=v.permute(0, 2, 1, 3).unsqueeze(2)
            if input_metadata.new_kv:
                input_metadata.k_new = input_metadata.k_new.permute(0, 2, 1, 3).unsqueeze(2)
                input_metadata.v_new = input_metadata.v_new.permute(0, 2, 1, 3).unsqueeze(2)


            input_metadata.layout = "bsghd"
        elif input_metadata.layout == "bsghd":
            pass
        elif input_metadata.layout is None:
            raise ValueError("Layout not given")

        assert input_metadata.layout == "bsghd"

        # get dims
        batch_size, seqlen_q, n_group_q, heads_per_group_q, dim_q = q.shape
        _, seqlen_k, n_group_k, heads_per_group_k, dim_k = k.shape
        _, seqlen_v, n_group_v, heads_per_group_v, dim_v = v.shape

        assert dim_q == dim_k == dim_v, f"Dimensions must match: {dim_q}, {dim_k}, {dim_v}"

        # get padded size
        dim_padded  = get_padded_headsize(dim_k)

        # Handle MQA/GQA case
        if heads_per_group_q > heads_per_group_k:
            input_metadata.is_gqa = True
        elif heads_per_group_q < heads_per_group_k:
            raise ValueError("heads_per_group_q < heads_per_group_k")
        else:
            input_metadata.is_gqa = False

        # context
        cls.SPLIT_K: Optional[int] = None
        cls.BLOCK_M = 16
        cls.BLOCK_N = 16

        cls.NUM_QUANT_GROUPS = 1  # Default quantization is row-wise

        # attn_bias = inp.attn_bias
        if input_metadata.cache_seqlens is not None:
            cache_seqlens = input_metadata.cache_seqlens
        else:
            cache_seqlens = None

        assert dim_k == dim_q, f"Keys have head dim {dim_k} but queries have head dim {dim_q}"

        BLOCK_M = cls.BLOCK_M
        BLOCK_N = cls.BLOCK_N
        if cls.SPLIT_K is not None:
            split_k = cls.SPLIT_K
        else:
            # Use heuristics
            split_k = get_split_k(batch_size, n_group_q, heads_per_group_q, seqlen_k) # NOTE: should the split think about seqlens?

        seqlen_q_ceil = (seqlen_q + BLOCK_M - 1) // BLOCK_M * BLOCK_M
        out_splitk = torch.empty([batch_size * n_group_q * heads_per_group_q, split_k, seqlen_q_ceil, dim_padded], dtype=torch.float32, device=q.device)
        metadata = torch.empty([batch_size * n_group_q * heads_per_group_q, 2, split_k, seqlen_q_ceil], dtype=torch.float32, device=q.device)
        lse = torch.empty((batch_size * n_group_q * heads_per_group_q, seqlen_q), device=q.device, dtype=torch.float32)
        grid = (triton.cdiv(seqlen_q, BLOCK_M), batch_size * n_group_q * heads_per_group_q, split_k)

        num_warps = 1
        split_size = (seqlen_k + split_k - 1) // split_k
        use_cache_seqlens = cache_seqlens is not None

        # Rotary Block Metadata (compute BLOCK_K)
        BLOCK_K = (
            32
            if input_metadata.rotary_dim <= 32
            else (64 if input_metadata.rotary_dim <= 64 else (128 if input_metadata.rotary_dim <= 128 else 256))
        )

        # import pdb; pdb.set_trace()

        # TODO: enable quantization
        _fwd_kernel_splitK[grid](
            Q=q,
            K=k,
            V=v,
            sm_scale=input_metadata.sm_scale,
            Out_splitK=out_splitk,
            Metadata=metadata,
            K_new = input_metadata.k_new,
            V_new = input_metadata.v_new,
            Cache_seqlens=cache_seqlens,
            Cache_batch_idx=input_metadata.cache_batch_idx,
            Alibi_slopes=input_metadata.alibi_slopes,
            Rotary_cos = input_metadata.rotary_cos,
            Rotary_sin = input_metadata.rotary_sin,
            Rotary_dim = input_metadata.rotary_dim,
            Rotary_interleaved = input_metadata.rotary_interleaved,
            Rotary_conjugate = input_metadata.rotary_conjugate,
            seqlen_ro = input_metadata.rotary_seqlen,
            IS_SEQLEN_OFFSETS_TENSOR = isinstance(cache_seqlens, torch.Tensor),
            IS_VARLEN = input_metadata.rotary_is_varlen,
            BLOCK_K=BLOCK_K,
            **_strides(q, "qz", "qm", "qg", "qh", "qd"),
            **_strides(k, "kz", "kn", "kg", "kh", "kd"),
            **_strides(v, "vz", "vn", "vg", "vh", "vd"),
            **_strides(out_splitk, "osk_zhg", "osk_s", "osk_m", "osk_d"),
            **_strides(metadata, "mzhg", "m2", "ms", "mm"),
            **_strides(input_metadata.k_new, "kn_z", "kn_n", "kn_g", "kn_h", "kn_d"),
            **_strides(input_metadata.v_new, "vn_z", "vn_n", "vn_g", "vn_h", "vn_d"),
            **_strides(input_metadata.alibi_slopes, "az", "ah"),
            Z=batch_size,
            H_q=heads_per_group_q,
            H_kv=heads_per_group_k,
            G_q=n_group_q,
            N_CTX_Q=seqlen_q,
            N_CTX_K=seqlen_k,
            N_CTX_NEW=input_metadata.k_new.shape[1] if input_metadata.new_kv else None,
            BLOCK_N_PER_SPLIT=split_size,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=dim_padded,
            ACTUAL_BLOCK_DMODEL=dim_k,
            BOUNDS_CHECKS_N=(split_size % BLOCK_N) > 0 or use_cache_seqlens,
            USE_CACHE_SEQLENS=use_cache_seqlens,
            USE_CACHE_BATCH_IDX= input_metadata.cache_batch_idx is not None,
            NEW_KV=input_metadata.new_kv,
            IS_GQA=input_metadata.is_gqa,
            IS_CAUSAL=input_metadata.causal,
            USE_ALIBI=False if input_metadata.alibi_slopes is None else True,
            USE_ROTARY=False if input_metadata.rotary_cos is None or input_metadata.rotary_sin is None else True,
            num_warps=num_warps,
            num_stages=1,
        )

        out = torch.empty((batch_size, seqlen_q, n_group_q, heads_per_group_q, dim_padded), device=q.device, dtype=q.dtype)

        # Merge together
        splitK_pow2 = triton.next_power_of_2(split_k)
        use_mask = splitK_pow2 > split_k
        if batch_size * n_group_q * heads_per_group_q * seqlen_q >= 512:
            k_block_num = 1
        else:
            k_block_num = 2
        assert dim_padded % k_block_num == 0
        k_block_size = dim_padded // k_block_num
        grid = (batch_size * n_group_q * heads_per_group_q, seqlen_q, k_block_num)

        _splitK_reduce[grid](
            out_splitk, 
            metadata, 
            out, 
            lse, 
            **_strides(out_splitk, "osk_zhg", "osk_s", "osk_m", "osk_k"),
            **_strides(metadata, "mzhg", "m2", "ms", "mm"), 
            **_strides(out, "oz", "om", "og", "oh", "ok"),
            **_strides(lse, "lse_zhg", "lse_m"), 
            M_ceil=seqlen_q_ceil, 
            BLOCK_SIZE=k_block_size, 
            G=n_group_q, 
            H=heads_per_group_q,
            # TODO: Tune num_warps
            split_k=split_k, 
            splitK_pow2=splitK_pow2, 
            use_mask=use_mask,
            IS_CAUSAL=input_metadata.causal,
            num_warps=4)

        lse = lse.reshape([batch_size, n_group_q, heads_per_group_q, seqlen_q])
        if q.ndim == 4:
            # BMGHK -> BMHK
            assert n_group_q == 1
            out = out[:, :, 0]
            lse = lse[:, 0]
        if seqlen_k == 0:
            out.zero_()
        out = out.reshape(batch_size, heads_per_group_q * n_group_q, -1, dim_padded).contiguous()

        # output is batch_size, heads_per_group_q * group_q, seqlen_q, dim_q
        if original_layout == "bshd":
            # out=out.transpose(1, 2).contiguous() # this screws up heads and data.
            # the data is laid out properly. Just need to reshape dims
            out = out.reshape(batch_size, seqlen_q, -1, dim_padded)

        return out.narrow(-1, 0, dim_k), lse


attention_decode = _attention.apply


def get_input_shapes():
    cases = [(max(1, 2**(16 - i)), 1, 2**i, 16, 1, 128)
             for i in range(8, 18)] + [(max(1, 2**(16 - i)), 1, 2**i, 16, 2, 128) for i in range(8, 18)]

    return cases


@pytest.mark.parametrize('batch_size, seqlen_q, seqlen_k, group_q, group_k, dim', get_input_shapes())
def test_op_fwd(batch_size, seqlen_q, seqlen_k, group_q, group_k, dim, dtype=torch.bfloat16):
    print()
    print(f"batch_size = {batch_size}, seqlen_q = {seqlen_q}, seqlen_k = {seqlen_k}, group_q = {group_q}, group_k = {group_k}, dim = {dim}")
    torch.manual_seed(20)
    query_group_head_size = (group_q + group_k - 1) // group_k
    q = (torch.empty((batch_size, seqlen_q, group_k, query_group_head_size, dim), dtype=dtype,
                     device="cuda").normal_(mean=0., std=0.5).requires_grad_())
    k = (torch.empty((batch_size, seqlen_k, group_k, 1, dim), dtype=dtype,
                     device="cuda").normal_(mean=0.,
                                            std=0.5).requires_grad_()).expand(-1, -1, -1, query_group_head_size, -1)
    v = (torch.empty((batch_size, seqlen_k, group_k, 1, dim), dtype=dtype,
                     device="cuda").normal_(mean=0.,
                                            std=0.5).requires_grad_()).expand(-1, -1, -1, query_group_head_size, -1)
    scale = 1 / dim**0.5
    input_metadata = MetaData(sm_scale=scale)
    input_metadata.layout = "bsghd"
    tri_out, _ = attention_decode(q, k, v, input_metadata)

    q = q.reshape([batch_size, seqlen_q, -1, dim]).permute(0, 2, 1, 3)
    k = k.reshape([batch_size, seqlen_k, -1, dim]).permute(0, 2, 1, 3)
    v = v.reshape([batch_size, seqlen_k, -1, dim]).permute(0, 2, 1, 3)
    attn = (q @ k.transpose(-1, -2) * scale).softmax(-1)
    ref_out = attn @ v

    # compare
    torch.testing.assert_close(ref_out, tri_out, atol=1e-3, rtol=0)


@pytest.mark.parametrize('B, Mq, Mkv, Hq, Hkv, K', get_input_shapes())
def test_op_fwd_int4_kv(B, Mq, Mkv, Hq, Hkv, K, dtype=torch.float16):
    torch.manual_seed(2)
    q = (torch.empty((B, Mq, Hkv, (Hq + Hkv - 1) // Hkv, K), dtype=dtype,
                     device="cuda").normal_(mean=1.0, std=0.5).requires_grad_())
    k = (torch.empty((B, Mkv, Hkv, 1, K), dtype=dtype,
                     device="cuda").normal_(mean=1.0,
                                            std=0.5).requires_grad_()).expand(-1, -1, -1, (Hq + Hkv - 1) // Hkv, -1)
    v = (torch.empty((B, Mkv, Hkv, 1, K), dtype=dtype,
                     device="cuda").normal_(mean=1.0,
                                            std=0.5).requires_grad_()).expand(-1, -1, -1, (Hq + Hkv - 1) // Hkv, -1)

    num_groups = 1
    quant_k = (quantize_kv_int4(k, num_groups=num_groups).contiguous().view(torch.int32))
    quant_v = (quantize_kv_int4(v, num_groups=num_groups).contiguous().view(torch.int32))
    scale = 1 / K**0.5
    input_metadata = MetaData(sm_scale=scale)
    input_metadata.layout = "bsghd"
    tri_out, _ = attention_decode(q, quant_k, quant_v, input_metadata)

    q = q.reshape([B, Mq, -1, K]).permute(0, 2, 1, 3)
    k = k.reshape([B, Mkv, -1, K]).permute(0, 2, 1, 3)
    v = v.reshape([B, Mkv, -1, K]).permute(0, 2, 1, 3)
    attn = (q @ k.transpose(-1, -2) * scale).softmax(-1)
    ref_out = attn @ v
    # compare
    torch.testing.assert_close(ref_out, tri_out, atol=2.1e-2, rtol=0)

    # since quantization introduces rounding error, use the
    # dequantized kv as inputs to the ref implementation to reduce
    # the tolerance to 1e-3
    dqk = dequantize_kv_fp16(quant_k, num_groups=num_groups)
    dqv = dequantize_kv_fp16(quant_v, num_groups=num_groups)
    dqk = dqk.reshape([B, Mkv, -1, K]).permute(0, 2, 1, 3)
    dqv = dqv.reshape([B, Mkv, -1, K]).permute(0, 2, 1, 3)
    dq_attn = (q @ dqk.transpose(-1, -2) * scale).softmax(-1)
    dq_ref_out = dq_attn @ dqv
    torch.testing.assert_close(dq_ref_out, tri_out, atol=1e-3, rtol=0)


def test_quantization():
    a = torch.randn((2, 4, 32), dtype=torch.float16, device='cuda')
    qa = quantize_kv_int4(a, num_groups=4)
    dqa = dequantize_kv_fp16(qa, num_groups=4)
    torch.testing.assert_close(a, dqa, atol=1.5e-1, rtol=1e-1)


try:
    FLASH_VER = 2
except BaseException:
    try:
        FLASH_VER = 1
    except BaseException:
        FLASH_VER = None
HAS_FLASH = FLASH_VER is not None

configs = []
for mode in ['fwd']:
    # for D_HEAD in [128]:
    for causal in [False]:
        configs.append(
            triton.testing.Benchmark(
                x_names=['B', 'Mq', 'Mkv', 'Hq', 'Hkv', 'K'], x_vals=get_input_shapes(), line_arg='provider',
                line_vals=['triton'] + (['flash'] if HAS_FLASH else []),
                line_names=['Triton'] + ([f'Flash-{FLASH_VER}'] if HAS_FLASH else []), styles=[('red', '-'),
                                                                                               ('blue', '-')],
                ylabel='ms', plot_name=f'fused-attention-d{128}-{mode}-causal={causal}', args={
                    # 'D_HEAD': D_HEAD,
                    'dtype': torch.float16, 'mode': mode, 'causal': causal
                }))


@triton.testing.perf_report(configs)
def bench_flash_attention(B, Mq, Mkv, Hq, Hkv, K, causal, mode, provider, dtype=torch.float16, device="cuda"):
    assert mode in ['fwd', 'bwd']
    warmup = 100
    rep = 400
    ms = 0
    if provider == "triton":
        q = torch.randn([B, Mq, Hkv, Hq // Hkv, K], device="cuda", dtype=dtype, requires_grad=False)
        k = torch.randn([B, Mkv, Hkv, 1, K], device="cuda", dtype=dtype,
                        requires_grad=False).expand(-1, -1, -1, Hq // Hkv, -1)
        v = torch.randn([B, Mkv, Hkv, 1, K], device="cuda", dtype=dtype,
                        requires_grad=False).expand(-1, -1, -1, Hq // Hkv, -1)

        sm_scale = 1.3
        input_metadata = MetaData(sm_scale=sm_scale)
        input_metadata.layout = "bsghd"
        fn = lambda: attention_decode(q, k, v, input_metadata)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    # flops_per_matmul = 2 * B * Hq * (Mq * K * Mkv + Mq * Mkv * K)
    # total_flops = 2 * flops_per_matmul
    # totalBytes = ((B * Mkv * Hkv * K * 2) + (B * Mq * Hq * K) + (B * Mq * Hq * K)) * 2

    # return totalBytes / ms * 1e-9
    return ms * 1000


def main():
    bench_flash_attention.run(save_path='.', print_data=True)


if __name__ == '__main__':
    sys.exit(main())
