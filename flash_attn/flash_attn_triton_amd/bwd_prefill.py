import torch
import triton
import triton.language as tl

from .bwd_ref import attention_backward_pytorch_ref_impl
from .utils import get_shape_from_layout, get_strides_from_layout

DEBUG = False


@triton.jit
def dropout_offsets(philox_seed, philox_offset, dropout_p, m, n, stride):
    ms = tl.arange(0, m)
    ns = tl.arange(0, n)
    return philox_offset + ms[:, None] * stride + ns[None, :]


@triton.jit
def dropout_rng(philox_seed, philox_offset, dropout_p, m, n, stride):
    rng_offsets = dropout_offsets(philox_seed, philox_offset, dropout_p, m, n, stride).to(tl.uint32)
    # TODO: use tl.randint for better performance
    return tl.rand(philox_seed, rng_offsets)


@triton.jit
def dropout_mask(philox_seed, philox_offset, dropout_p, m, n, stride):
    rng_output = dropout_rng(philox_seed, philox_offset, dropout_p, m, n, stride)
    rng_keep = rng_output > dropout_p
    return rng_keep

@triton.jit
def store_dropout_mask(X, philox_seed, philox_offset, dropout_p: tl.constexpr, m: tl.constexpr, n: tl.constexpr, stride: tl.constexpr):
    x = tl.zeros((m, n), tl.float32)
    # import pdb; pdb.set_trace()
    x = dropout_mask(philox_seed, philox_offset, dropout_p, m, n, stride)
    x_block = (tl.arange(0, m)[:, None]*n + tl.arange(0, n)[None, :])
    tl.store(X+x_block, x, mask=((tl.arange(0, m)[:, None] < m) & (tl.arange(0, n)[None, :] < n)))


@triton.jit
def _bwd_preprocess_use_o(
    Out,
    DO,
    Delta,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_doz, stride_doh, stride_dom, stride_dok,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    ACTUAL_BLOCK_DMODEL: tl.constexpr,
    N_CTX_Q: tl.constexpr,
    Z: tl.constexpr,
    H: tl.constexpr,
    IS_VARLEN: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # Compute batch and head indices
    off_z = pid_bh // H
    off_h = pid_bh % H

    if IS_VARLEN:
        # Compute sequence lengths for the current batch
        q_start = tl.load(cu_seqlens_q + off_z)
        q_end = tl.load(cu_seqlens_q + off_z + 1)
        k_start = tl.load(cu_seqlens_k + off_z)
        k_end = tl.load(cu_seqlens_k + off_z + 1)

        # Compute actual sequence lengths
        N_CTX_Q = q_end - q_start
        N_CTX_K = k_end - k_start
    else:
        q_start = 0
        k_start = 0
        N_CTX_Q = max_seqlen_q
        N_CTX_K = max_seqlen_k

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, BLOCK_DMODEL)

    # create masks
    mask_m = off_m < N_CTX_Q
    mask_d = off_d < ACTUAL_BLOCK_DMODEL

    # compute offsets
    o_offset = Out + off_z * stride_oz + off_h * stride_oh + q_start * stride_om
    do_offset = DO + off_z * stride_oz + off_h * stride_oh + q_start * stride_om

    # compute pointers
    out_ptrs = o_offset + off_m[:, None] * stride_om + off_d[None, :] * stride_ok
    do_ptrs = do_offset + off_m[:, None] * stride_dom + off_d[None, :] * stride_dok

    # load
    o = tl.load(out_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    # compute delta
    delta = tl.sum(o * do, axis=1)

    # write-back delta
    if IS_VARLEN:
        delta_offset = tl.load(cu_seqlens_q + off_z) * H + off_h * N_CTX_Q + off_m
        delta_ptrs = Delta + delta_offset
    else:
        delta_ptrs = Delta + pid_bh * N_CTX_Q + off_m
    tl.store(delta_ptrs, delta, mask=mask_m)


@triton.jit
def _bwd_preprocess_use_p(
    Q,       # Pointer to queries
    K,       # Pointer to keys
    V,       # Pointer to values
    DO,      # Pointer to gradients of the output
    LSE,     # Pointer to log-sum-exp from forward pass
    Delta,   # Pointer to store delta
    sm_scale,    # Softmax scaling factor
    stride_dq_all,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    N_CTX_Q: tl.constexpr,  # Number of query tokens
    N_CTX_K: tl.constexpr,  # Number of key tokens
    BLOCK_M: tl.constexpr,  # Block size for M dimension
    BLOCK_N: tl.constexpr,  # Block size for N dimension
    BLOCK_DMODEL: tl.constexpr,  # Block size for model dimension
    ACTUAL_BLOCK_DMODEL: tl.constexpr,
    USE_EXP2: tl.constexpr,      # Whether to use exp2 for exponentials
    Z: tl.constexpr,             # Batch size
    H: tl.constexpr,             # Number of heads
):
    # Compute program IDs for blocks
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # compute batch and head indices
    batch_idx = pid_bh // H
    head_idx = pid_bh % H

    # Compute offsets for M and D dimensions
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, BLOCK_DMODEL)

    # Create masks for bounds checking
    mask_m = off_m < N_CTX_Q
    mask_d = off_d < ACTUAL_BLOCK_DMODEL

    # Compute pointers for Q and DO
    q_ptrs = Q + batch_idx * stride_qz + head_idx * stride_qh + off_m[:, None] * stride_qm + off_d[None, :] * stride_qk
    do_ptrs = DO + batch_idx * stride_qz + head_idx * stride_qh + off_m[:, None] * stride_qm + off_d[None, :] * stride_qk

    # Load Q and DO
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    # Initialize delta accumulator
    delta = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Loop over blocks of K and V
    for start_n in range(0, N_CTX_K, BLOCK_N):
        # Compute offsets for N dimension
        off_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = off_n < N_CTX_K

        # Compute pointers for K and V
        k_ptrs = K + batch_idx * stride_kz + head_idx * stride_kh + off_n[:, None] * stride_kn + off_d[None, :] * stride_kk
        v_ptrs = V + batch_idx * stride_vz + head_idx * stride_vh + off_n[:, None] * stride_vn + off_d[None, :] * stride_vk

        # Load K and V
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

        # Compute QK^T
        qk = tl.dot(q, tl.trans(k))

        # Load LSE
        lse_ptrs = LSE + pid_bh * N_CTX_Q + off_m
        lse = tl.load(lse_ptrs, mask=mask_m, other=float('-inf'))

        # Compute scaled QK and softmax probabilities
        if USE_EXP2:
            RCP_LN2: tl.constexpr = 1.4426950408889634
            qk_scaled = qk * sm_scale * RCP_LN2
            lse_scaled = lse * RCP_LN2
            p = tl.exp2(qk_scaled - lse_scaled[:, None])
        else:
            qk_scaled = qk * sm_scale
            p = tl.exp(qk_scaled - lse[:, None])

        # Mask p where necessary
        p = tl.where(mask_m[:, None] & mask_n[None, :], p, 0.0)

        # Compute dp = DO @ V^T
        v_t = tl.trans(v)
        dp = tl.dot(do, v_t)

        # Accumulate delta
        delta += tl.sum(p * dp, axis=1)

    # Write-back delta
    delta_ptrs = Delta + pid_bh * N_CTX_Q + off_m
    tl.store(delta_ptrs, delta, mask=mask_m)


@triton.jit
def _bwd_kernel_one_col_block(
        Q,
        K,
        V,
        sm_scale,
        Out,
        DO,
        DQ,
        DK,
        DV,
        L,
        D,
        q_offset,
        k_offset,
        v_offset,
        do_offset,
        dq_offset,
        dk_offset,
        dv_offset,
        d_offset,
        l_offset,
        stride_dq_all,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vn,
        stride_vk,
        Z,
        H,
        N_CTX_Q,
        N_CTX_K,
        off_h,
        off_z,
        off_hz,
        start_n,
        num_block_m,
        num_block_n,
        dropout_p, philox_seed, philox_offset_base,
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        ACTUAL_BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
        SEQUENCE_PARALLEL: tl.constexpr,
        CAUSAL: tl.constexpr,
        USE_EXP2: tl.constexpr,
        ENABLE_DROPOUT: tl.constexpr,          
):
    if CAUSAL:
        lo = start_n * BLOCK_M
    else:
        lo = 0

    # initialize col and head offsets
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # masks
    mask_n = offs_n < N_CTX_K
    mask_d = offs_d < ACTUAL_BLOCK_DMODEL
    k_mask = mask_n[:, None] & mask_d[None, :]
    v_mask = mask_n[:, None] & mask_d[None, :]
    

    # initialize grad accumulators
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    # load k and v once per column block
    k_ptrs = k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    k = tl.load(k_ptrs, mask=k_mask, other=0.0)
    v = tl.load(v_ptrs, mask=v_mask, other=0.0)
    # print("k:", k)
    # print("v:", v)

    # loop over rows
    for start_m in range(lo, num_block_m * BLOCK_M, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        q_ptrs = q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        dq_ptrs = dq_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        do_ptrs = do_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        
        # update mask as row block changes
        mask_m = offs_m < N_CTX_Q
        q_mask = mask_m[:, None] & mask_d[None, :]

        # load q, k, v, do on-chip
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)
        do = tl.load(do_ptrs, mask=q_mask, other=0.0)
        # print("q:", q)
        # print("do:", do)

        # recompute p = softmax(qk, dim=-1).T
        # NOTE: `do` is pre-divided by `l`; no normalization here
        if CAUSAL:
            qk = tl.where(
                offs_m[:, None] >= offs_n[None, :], float(0.0), float("-inf")
            )
        else:
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        # print("qk:", qk)
        l_ptrs = l_offset + offs_m
        l_i = tl.load(l_ptrs, mask=mask_m)
        # print("l_i:", l_i)

        # compute p
        if USE_EXP2:
            RCP_LN2: tl.constexpr = 1.4426950408889634
            qk *= sm_scale * RCP_LN2
            l_i *= RCP_LN2
            p = tl.math.exp2(qk - l_i[:, None])
        else:
            qk *= sm_scale
            p = tl.math.exp(qk - l_i[:, None])
        # print("p:", p)
        # mask block in the cases where the data is smaller the block size
        p_mask = mask_m[:, None] & mask_n[None, :]
        p = tl.where(p_mask, p, 0.0)
        # print("p masked:", p)
        
        # compute dv
        dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)
        # print("dv:", dv)

        # compute dp
        dp = tl.dot(do, tl.trans(v))
        # print("dp:", dp)

        # compute ds , ds = p * (dp - delta[:, None])
        if True:
            d_ptrs = d_offset + offs_m
            Di = tl.load(d_ptrs, mask=mask_m)
            ds = (p * (dp - Di[:, None])) * sm_scale
        else:
            delta = tl.sum(p * dp, axis=1)
            delta = tl.where(mask_m, delta, 0.0)
            ds = (p * (dp - delta[:, None])) * sm_scale
        # print("ds:", ds)
        ds = tl.where(p_mask, ds, 0.0).to(Q.dtype.element_ty)
        # print("ds masked:", ds)

        # if dropout enabled, mask the scores and scale proportionally
        # print('ds_before_triton\n', ds)
        if ENABLE_DROPOUT:
            philox_offset = philox_offset_base + start_m * N_CTX_K + start_n * BLOCK_N
            # import pdb; pdb.set_trace()
            keep = dropout_mask(philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, N_CTX_K)
            ds = tl.where(keep, ds, 0.0)

            ds = ds / (1 - dropout_p) # scale ds based on dropout_p
            ds = ds.to(Q.dtype.element_ty)
        # print('ds_after_triton\n', ds)

        # compute dk = dot(ds.T, q)
        dk += tl.dot(tl.trans(ds), q)
        # print("dk:", dk)

        # compute dq
        if SEQUENCE_PARALLEL:
            if True: # path for MMA_V3 in oai kernel
                dq = tl.dot(ds, k)
            else:
                # not work with mma v3, because M % 64 != 0
                dq = tl.trans(tl.dot(tl.trans(k), tl.trans(ds)))
            tl.store(dq_ptrs, dq.to(Q.dtype.element_ty), mask=q_mask)
        else:
            dq = tl.load(dq_ptrs, mask=q_mask, other=0.0)
            # print("dq load:", dq)
            dq += tl.dot(ds, k)
            # print("dq after dot:", dq)
            tl.store(dq_ptrs, dq.to(Q.dtype.element_ty), mask=q_mask)

    # write-back dv and dk
    dk_ptrs = dk_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    dv_ptrs = dv_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    # write-back
    # print("dv:", dv)
    tl.store(dk_ptrs, dk.to(K.dtype.element_ty), mask=k_mask)
    tl.store(dv_ptrs, dv.to(V.dtype.element_ty), mask=v_mask)

# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'waves_per_eu': 1, 'PRE_LOAD_V': False}, num_stages=1,
#                       num_warps=4),
#     ],
#     key=['IS_CAUSAL', 'dropout_p', 'BLOCK_DMODEL'],
#     use_cuda_graph=True,
# )
@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    Out,
    DO,
    DQ,
    DK,
    DV,
    L,
    D,
    stride_dq_all,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    Z,
    H,
    dropout_p, philox_seed, philox_offset_base,
    num_block_m,
    num_block_n,
    cu_seqlens_q,  
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    ACTUAL_BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    CAUSAL: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
):
    # program ids
    off_hz = tl.program_id(0)
    if SEQUENCE_PARALLEL:
        start_n = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    if IS_VARLEN:
        # Compute sequence lengths for the current batch
        q_start = tl.load(cu_seqlens_q + off_z)
        q_end = tl.load(cu_seqlens_q + off_z + 1)
        k_start = tl.load(cu_seqlens_k + off_z)
        k_end = tl.load(cu_seqlens_k + off_z + 1)

        # Compute actual sequence lengths
        N_CTX_Q = q_end - q_start
        N_CTX_K = k_end - k_start
    else:
        q_start = 0
        k_start = 0
        N_CTX_Q = max_seqlen_q
        N_CTX_K = max_seqlen_k
    

    # input tensor offsets
    q_offset = Q + off_z * stride_qz + off_h * stride_qh + q_start * stride_qm
    k_offset = K + off_z * stride_kz + off_h * stride_kh + k_start * stride_kn
    v_offset = V + off_z * stride_vz + off_h * stride_vh + k_start * stride_vn
    do_offset = DO + off_z * stride_qz + off_h * stride_qh + q_start * stride_qm
    if IS_VARLEN:
        delta_offset = tl.load(cu_seqlens_q + off_z) * H + off_h * N_CTX_Q
        l_offset = L + delta_offset
        d_offset = D + delta_offset
    else:
        l_offset = L + off_hz * N_CTX_Q 
        d_offset = D + off_hz * N_CTX_Q

    # output tensor offsets
    dk_offset = DK + off_z * stride_kz + off_h * stride_kh + k_start * stride_kn
    dv_offset = DV + off_z * stride_vz + off_h * stride_vh + k_start * stride_vn
    if SEQUENCE_PARALLEL:
        dq_offset = DQ + stride_dq_all * start_n + off_z * stride_qz + off_h * stride_qh + q_start * stride_qm
    else:
        dq_offset = DQ + off_z * stride_qz + off_h * stride_qh + q_start * stride_qm

    # inner loop
    if SEQUENCE_PARALLEL:
        _bwd_kernel_one_col_block(
            Q,
            K,
            V,
            sm_scale,
            Out,
            DO,
            DQ,
            DK,
            DV,
            L,
            D,
            q_offset,
            k_offset,
            v_offset,
            do_offset,
            dq_offset,
            dk_offset,
            dv_offset,
            d_offset,
            l_offset,
            stride_dq_all,
            stride_qz,
            stride_qh,
            stride_qm,
            stride_qk,
            stride_kz,
            stride_kh,
            stride_kn,
            stride_kk,
            stride_vz,
            stride_vh,
            stride_vn,
            stride_vk,
            Z,
            H,
            N_CTX_Q,
            N_CTX_K,
            off_h,
            off_z,
            off_hz,
            start_n,
            num_block_m,
            num_block_n,
            dropout_p, philox_seed, philox_offset_base,
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL=BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL=ACTUAL_BLOCK_DMODEL,
            BLOCK_N=BLOCK_N,
            SEQUENCE_PARALLEL=SEQUENCE_PARALLEL,
            CAUSAL=CAUSAL,
            USE_EXP2=USE_EXP2,
            ENABLE_DROPOUT=ENABLE_DROPOUT,          
        )
    else:
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(
                Q,
                K,
                V,
                sm_scale,
                Out,
                DO,
                DQ,
                DK,
                DV,
                L,
                D,
                q_offset,
                k_offset,
                v_offset,
                do_offset,
                dq_offset,
                dk_offset,
                dv_offset,
                d_offset,
                l_offset,
                stride_dq_all,
                stride_qz,
                stride_qh,
                stride_qm,
                stride_qk,
                stride_kz,
                stride_kh,
                stride_kn,
                stride_kk,
                stride_vz,
                stride_vh,
                stride_vn,
                stride_vk,
                Z,
                H,
                N_CTX_Q,
                N_CTX_K,
                off_h,
                off_z,
                off_hz,
                start_n,
                num_block_m,
                num_block_n,
                dropout_p, philox_seed, philox_offset_base,
                BLOCK_M=BLOCK_M,
                BLOCK_DMODEL=BLOCK_DMODEL,
                ACTUAL_BLOCK_DMODEL=ACTUAL_BLOCK_DMODEL,
                BLOCK_N=BLOCK_N,
                SEQUENCE_PARALLEL=SEQUENCE_PARALLEL,
                CAUSAL=CAUSAL,
                USE_EXP2=USE_EXP2,
                ENABLE_DROPOUT=ENABLE_DROPOUT,                
            )


# NOTE: smaller blocks have lower accuracy. more accumlation error probably 128 * 128 seems good but leads to oom. 64 * 64 has accumlation errors but no oom.
def attention_prefill_backward_triton_new_impl(
    do,
    q,
    k,
    v,
    o,
    softmax_lse,
    dq,
    dk,
    dv,
    sm_scale: float,
    alibi_slopes,
    causal,
    layout: str,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p,
    dropout_philox_seed,
    dropout_philox_offset,
    use_exp2: bool,
    bwd_preprocessing_use_o: bool,
    BLOCK_M=64,
    BLOCK_N=64,
):
    if DEBUG:
        print()
        print("attention_prefill_backward_triton_new_impl")
        print("do:", do, do.shape)
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("o:", o, o.shape)
        print("softmax_lse:", softmax_lse, softmax_lse.shape)
        print("dq:", dq, dq.shape if dq is not None else None)
        print("dk:", dk, dk.shape if dk is not None else None)
        print("dv:", dv, dv.shape if dv is not None else None)
        print("sm_scale:", sm_scale)
        print("alibi_slopes:", alibi_slopes)
        print("layout:", layout)        
        print("cu_seqlens_q:", cu_seqlens_q)
        print("cu_seqlens_k:", cu_seqlens_k)
        print("max_seqlen_q:", max_seqlen_q)
        print("max_seqlen_k:", max_seqlen_k)
        print("dropout_p:", dropout_p)
        print("dropout_philox_seed:", dropout_philox_seed)
        print("dropout_philox_offset:", dropout_philox_offset)
        print("use_exp2:", use_exp2)
        print("bwd_preprocessing_use_o:", bwd_preprocessing_use_o)
        print("BLOCK_M:", BLOCK_M)
        print("BLOCK_N:", BLOCK_N)

    # make contigious
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    softmax_lse = softmax_lse.contiguous()

    # get strides and shape
    if True:
        batch, nheads_q, nheads_k, head_size, max_seqlen_q, max_seqlen_k = get_shape_from_layout(q, k, layout, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
        q_strides, k_strides, v_strides, o_strides = get_strides_from_layout(q, k, v, o, layout)
        stride_qz, stride_qh, stride_qm, stride_qk =  q_strides
        stride_kz, stride_kh, stride_kn, stride_kk = k_strides
        stride_vz, stride_vh, stride_vn, stride_vk = v_strides
        stride_oz, stride_oh, stride_om, stride_ok = o_strides
        stride_dq_all = q.numel()
        batch_headsize = batch * nheads_q
    else:
        batch_q, heads_q, seqlen_q, head_size_q = q.shape
        batch_k, heads_k, seqlen_k, head_size_k = k.shape
        batch_headsize = batch_q * heads_q
        stride_dq_all = dq.numel()
        stride_qz, stride_qh, stride_qm, stride_qk =  q.stride(0),  q.stride(1), q.stride(2),  q.stride(3)
        stride_kz, stride_kh, stride_kn, stride_kk = k.stride(0),  k.stride(1), k.stride(2),  k.stride(3)
        stride_vz, stride_vh, stride_vn, stride_vk = v.stride(0),  v.stride(1), v.stride(2),  v.stride(3)

    sequence_parallel = False
    causal = False

    # divide up the problem
    num_blocks_m = triton.cdiv(max_seqlen_q, BLOCK_M)
    num_blocks_n = triton.cdiv(max_seqlen_k, BLOCK_N)

    # get closest power of 2 over or equal to 32.
    padded_d_model = 1 << (head_size - 1).bit_length()
    padded_d_model = max(padded_d_model, 16)
    BLOCK_DMODEL = padded_d_model
    ACTUAL_BLOCK_DMODEL = head_size

    do = do.contiguous()
    if sequence_parallel:
        # replicate q for each parallel sequence
        replicas = num_blocks_n
        new_dq_shape = (replicas,) + q.shape
        if dq is None: 
            dq = torch.zeros(new_dq_shape, device=q.device, dtype=q.dtype)
        else:
            dq = dq.contiguous()
    else:
        if dq is None:
            dq = torch.zeros_like(q, dtype=q.dtype)
        else:
            dq = dq.contiguous()

    # NOTE: the kernel does inplace accumlation s  o dq has to be zeros. This avoids the case where we are passed empty dq and it is not all zeros
    dq.zero_()

    if dk is None:
        if True:
            dk = torch.zeros_like(k)
        else:
            dk = torch.empty_like(k)
    else:
        dk = dk.contiguous()

    if dv is None:
        if True:
            dv = torch.zeros_like(v)
        else:
            dv = torch.empty_like(v)
    else:
        dv = dv.contiguous()

    # assert contigious
    assert do.is_contiguous()
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert o.is_contiguous()
    assert softmax_lse.is_contiguous()
    assert dq.is_contiguous()
    assert dk.is_contiguous()
    assert dv.is_contiguous()

    num_warps = 4 # NOTE: originial is 8. changing it to 1 caused issues be careful
    num_stages = 1

    if True:
        delta = torch.zeros_like(softmax_lse)
    else:
        delta = torch.empty_like(softmax_lse)

    is_varlen = layout == "thd"

    if bwd_preprocessing_use_o:
        _bwd_preprocess_use_o[(num_blocks_m, batch_headsize)](
            o,
            do,
            delta,
            stride_oz, stride_oh, stride_om, stride_ok,
            stride_oz, stride_oh, stride_om, stride_ok,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL=BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL=ACTUAL_BLOCK_DMODEL,
            N_CTX_Q=max_seqlen_q,
            Z=batch,
            H=nheads_q,
            IS_VARLEN=is_varlen
        )
    else:
        _bwd_preprocess_use_p[(num_blocks_m, batch_headsize)](
            q,
            k,
            v,
            do,
            softmax_lse,
            delta,
            sm_scale,
            stride_dq_all,
            stride_qz,
            stride_qh,
            stride_qm,
            stride_qk,
            stride_kz,
            stride_kh,
            stride_kn,
            stride_kk,
            stride_vz,
            stride_vh,
            stride_vn,
            stride_vk,
            Z=batch,
            H=nheads_q,
            N_CTX_Q=max_seqlen_q,
            N_CTX_K=max_seqlen_k,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL=ACTUAL_BLOCK_DMODEL,
            USE_EXP2=use_exp2,
        )

    if DEBUG:
        print("_bwd_kernel inputs")
        print("do:", do, do.shape)
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("sm_scale", sm_scale)
        print("o:", o, o.shape)
        print("dq:", dq, dq.shape)
        print("dk:", dk, dk.shape)
        print("dv:", dv, dv.shape)
        print("L:", softmax_lse, softmax_lse.shape)
        print("delta:", delta, delta.shape)
        print("stride_qz, stride_qh, stride_qm, stride_qk:",  stride_qz, stride_qh, stride_qm, stride_qk)
        print("stride_kz, stride_kh, stride_kn, stride_kk:",  stride_kz, stride_kh, stride_kn, stride_kk)
        print("stride_vz, stride_vh, stride_vn, stride_vk:",  stride_vz, stride_vh, stride_vn, stride_vk)
        print("batch_q:", batch)
        print("heads_q:",nheads_q)
        print("max_seqlen_q:",max_seqlen_q)
        print("max_seqlen_k:",max_seqlen_k)
        print("BLOCK_M:",BLOCK_M)
        print("BLOCK_N:",BLOCK_M)
        print("BLOCK_DMODEL:",BLOCK_DMODEL)
        print("ACTUAL_BLOCK_DMODEL:",ACTUAL_BLOCK_DMODEL)
        print("SEQUENCE_PARALLEL:",sequence_parallel)
        print("CAUSAL:",causal)
        print("num_warps:",num_warps)
        print("num_stages:", num_stages)
        print("USE_EXP2:", use_exp2)

    _bwd_kernel[(batch_headsize, num_blocks_n if sequence_parallel else 1)](
        q,
        k,
        v,
        sm_scale,
        o,
        do,
        dq,
        dk,
        dv,
        softmax_lse,
        delta,
        stride_dq_all,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        batch,
        nheads_q,
        dropout_p,
        dropout_philox_seed,
        dropout_philox_offset,
        num_blocks_m,
        num_blocks_n,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        ACTUAL_BLOCK_DMODEL=ACTUAL_BLOCK_DMODEL,
        SEQUENCE_PARALLEL=sequence_parallel,
        CAUSAL=causal,
        USE_EXP2=use_exp2,
        num_warps=num_warps,
        num_stages=num_stages,
        IS_VARLEN=is_varlen,
        ENABLE_DROPOUT=dropout_p >= 0.0,
    )

    if len(dq.shape) == 5:
        dq = dq.sum(dim=0)

    return dq, dk, dv, delta, None, None


def attention_prefill_backward_triton_impl(
    do,
    q,
    k,
    v,
    o,
    softmax_lse,
    dq,
    dk,
    dv,
    sm_scale: float,
    alibi_slopes,
    causal,
    layout: str,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p,
    dropout_philox_seed,
    dropout_philox_offset,
    use_exp2: bool,
    bwd_preprocessing_use_o: bool,
    use_new,
):
    if use_new:
        return attention_prefill_backward_triton_new_impl(
            do,
            q,
            k,
            v,
            o,
            softmax_lse,
            dq,
            dk,
            dv,
            sm_scale,
            alibi_slopes,
            causal,
            layout,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            dropout_philox_seed,
            dropout_philox_offset,
            use_exp2,
            bwd_preprocessing_use_o,
        )
    else:
        # test pytorch impl
        
        dropout_mask_ = torch.zeros(max_seqlen_q, max_seqlen_k)
        store_dropout_mask[(1, 1, 1)](MASK=dropout_mask_, philox_seed=dropout_philox_seed, philox_offset=dropout_philox_offset, dropout_p=dropout_p, m=max_seqlen_q, n=max_seqlen_k, stride=max_seqlen_k)
        print('dropout_mask', dropout_mask_)
        
        
        dq_ref, dk_ref, dv_ref, delta_ref = attention_backward_pytorch_ref_impl(
            do, q, k, v, o, softmax_lse, sm_scale, causal, dropout_mask_, dropout_p, layout, use_exp2, bwd_preprocessing_use_o
        )
        if dq is not None:
            dq.copy_(dq_ref)
        else:
            dq = dq_ref

        if dk is not None:
            dk.copy_(dk_ref)
        else:
            dk = dk_ref

        if dv is not None:
            dv.copy_(dv_ref)
        else:
            dv = dv_ref

        return dq, dk, dv, delta_ref, None, None
