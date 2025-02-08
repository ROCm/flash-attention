import torch
import triton # type: ignore
import triton.language as tl # type: ignore
from .utils import DROPOUT_USE_PYTORCH, DROPOUT_DUMP, get_shape_from_layout, \
    get_strides_from_layout, create_dropout_mask, create_dropout_mask_varlen
from bwd_prefill_split import _bwd_preprocess, _bwd_kernel_dkdv, _bwd_kernel_dq

# NOTE: triton fails to import tl.constexprs so create them here for the file
tl_DROPOUT_USE_PYTORCH: tl.constexpr = DROPOUT_USE_PYTORCH
tl_DROPOUT_DUMP: tl.constexpr = DROPOUT_DUMP


# grid = (tl.cdiv(max_seqlen_q // BLOCK_M2), batch, nheads_q)
@triton.jit
def bwd_kernel(
    Q, K, V, sm_scale, DO, DQ, DK, DV,
    M, Delta,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_dqb, stride_dqh, stride_dqm, stride_dqk,
    stride_dkb, stride_dkh, stride_dkn, stride_dkk,
    stride_deltab, stride_deltah, stride_deltam,
    stride_dob, stride_doh, stride_dom, stride_dok,
    stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn,
    HQ, HK,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    dropout_mask, dropout_p, philox_seed, philox_offset_base,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ACTUAL_HEAD_DIM: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_EXP2: tl.constexpr,
    DEBUG_TRITON: tl.constexpr,
    DEBUG_TRITON_DETAIL: tl.constexpr,
):
    # program ids
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    hkid = tl.program_id(2)
    # figure out varlen start and end
    q_start = 0
    k_start = 0
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k
    if IS_VARLEN:
        # Compute actual sequence lengths
        q_start = tl.load(cu_seqlens_q + bid)
        q_end = tl.load(cu_seqlens_q + bid + 1)
        k_start = tl.load(cu_seqlens_k + bid)
        k_end = tl.load(cu_seqlens_k + bid + 1)
        seqlen_q = q_end - q_start
        seqlen_k = k_end - k_start

    # align the delta_qk
    start_n = pid * BLOCK_N1
    if (start_n < seqlen_k):
        # This section does dk and dv
        dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

        delta_qk = seqlen_q - seqlen_k
        if DEBUG_TRITON: print(f"\npid: {pid}, bid: {bid}, hkid: {hkid}")  # noqa: E701
        if DEBUG_TRITON: print(f"delta_qk = {delta_qk}")  # noqa: E701
        # q > k: diretcly skip all the way until the start of causal block
        start_delta_q_gt_k = delta_qk
        # q < k: some blocks will have no Masked block, other needs to re-calc
        # starting position
        # delta_qk is negative so flip it, only multiple of BLOCK_N can skip the
        # masked op
        num_blocks_skip = -delta_qk // BLOCK_N1
        delta_aligned = (num_blocks_skip + 1) * BLOCK_N1 + delta_qk
        start_delta_q_lt_k = delta_aligned // BLOCK_M1 * BLOCK_M1
        if delta_qk >= 0:
            start_delta = delta_qk
            if DEBUG_TRITON: print(f"q >= k: start_delta = delta_qk aligned to BLOCK_M = {start_delta_q_gt_k}")  # noqa: E701
        else:
            start_delta = start_delta_q_lt_k
            if DEBUG_TRITON: print(f"q < k: start_delta = residue btw multiple BLOCK_N and delta_qk = {delta_aligned} = aligned to BLOCK_M = {start_delta_q_lt_k}")  # noqa: E701

        offs_k = tl.arange(0, HEAD_DIM)
        offs_n = start_n + tl.arange(0, BLOCK_N1)
        # Mask for loading K and V
        mask_kv = offs_n[:, None] < seqlen_k
        PADDED_HEAD: tl.constexpr = (ACTUAL_HEAD_DIM != HEAD_DIM)
        if PADDED_HEAD:
            mask_k = offs_k < ACTUAL_HEAD_DIM
            mask_kv &= mask_k[None, :]
        offs_kv = offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk

        GROUP_SIZE = HQ // HK
        # K/V tensors not changed for the group
        adj_kv = bid * stride_kb + hkid * stride_kh + k_start * stride_kn
        # load K and V: they stay in SRAM throughout the inner loop.
        k = tl.load(K + adj_kv + offs_kv, mask=mask_kv, other=0.0)
        v = tl.load(V + adj_kv + offs_kv, mask=mask_kv, other=0.0)
        # If MQA / GQA, set the K and V head offsets appropriately.
        # hqid = hkid
        for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
            if delta_qk >= 0:
                start_m = start_n + start_delta
                len_m = BLOCK_N1
            else:
                start_m = max(start_n + delta_qk, 0)
                start_m = start_m // BLOCK_M1 * BLOCK_M1
                # because we might shift the masked blocks up, we are deeper into
                # the masked out region, so we would potentially increase the total
                # steps with masked operation to get out of it
                residue_m = max(start_n + delta_qk - start_m, 0)
                len_m = BLOCK_N1 + residue_m
                if DEBUG_TRITON: print(f"residue_m = {residue_m}")  # noqa: E701

            # offset input and output tensor by batch and Q/K heads
            adj_q = bid * stride_qb + hqid * stride_qh + q_start * stride_qm
            Q_ptr = Q + adj_q
            adj_do = bid * stride_dob + hqid * stride_doh + q_start * stride_dom
            DO_ptr = DO + adj_do
            adj_delta = bid * stride_deltab + hqid * stride_deltah + \
                q_start * stride_deltam
            M_ptr = M + adj_delta
            Delta_ptr = Delta + adj_delta

            # batch_philox_offset is the ACTUALLY dropout offset
            # dropout_offset is for debug purpose and will be removed later
            batch_philox_offset = 0
            dropout_offset = 0
            if ENABLE_DROPOUT:
                batch_philox_offset = philox_offset_base + bid * stride_dropoutb + \
                                    hqid * stride_dropouth
                dropout_offset = dropout_mask + bid * stride_dropoutb + \
                                hqid * stride_dropouth

            MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
            # bound the masked operation to q len so it does not have to wast cycles
            len_m = min(len_m, seqlen_q)
            num_steps = tl.cdiv(len_m, MASK_BLOCK_M1)
            # when q < k, we may skip the initial masked op
            if pid < num_blocks_skip:
                num_steps = 0

            # if start_m is negative, the current N-tile has no block on the
            #   diagonal of causal mask, so everything have no causal mask
            if DEBUG_TRITON: print(f"Masked: start_n: {start_n}; start_m: {start_m}, num_steps: {num_steps}")  # noqa: E701
            dk, dv = _bwd_kernel_dkdv(
                dk, dv,  # output tensors
                Q_ptr, k, v, DO_ptr, M_ptr, Delta_ptr, sm_scale, # input tensors
                stride_qm, stride_qk,  # strides for q
                stride_dom, stride_dok,  # strides for o
                stride_dropoutm, stride_dropoutn,  # strides for dropout
                stride_deltam,
                MASK_BLOCK_M1, BLOCK_N1,  # block dim
                HEAD_DIM, ACTUAL_HEAD_DIM,  # head dim
                dropout_p, philox_seed, batch_philox_offset, dropout_offset,  #
                seqlen_q, seqlen_k,  # max sequence length for q and k
                start_n, start_m, num_steps,  # iteration numbers
                MASK=True,  # causal masking
                ENABLE_DROPOUT=ENABLE_DROPOUT,  # activate dropout
                USE_EXP2=USE_EXP2,
                DEBUG_TRITON=DEBUG_TRITON,
                DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
            )
            start_m += num_steps * MASK_BLOCK_M1
            num_steps = tl.cdiv(seqlen_q - start_m, BLOCK_M1)
            end_m = start_m + num_steps * BLOCK_M1

            if DEBUG_TRITON: print(f"start_m after Masked step: {start_m}; num_steps: {num_steps}")  # noqa: E701
            if DEBUG_TRITON: print(f"unMasked: start_n: {start_n}, start_m: {start_m}, end_m: {end_m}, num_steps: {num_steps}")  # noqa: E701
            if DEBUG_TRITON: print("unMasked")  # noqa: E701
            dk, dv = _bwd_kernel_dkdv(
                dk, dv,  # output tensors
                Q_ptr, k, v, DO_ptr, M_ptr, Delta_ptr, sm_scale, # input tensors
                stride_qm, stride_qk,  # strides for q
                stride_dom, stride_dok,  # strides for o
                stride_dropoutm, stride_dropoutn,  # strides for dropout
                stride_deltam,
                BLOCK_M1, BLOCK_N1,  # block dim
                HEAD_DIM, ACTUAL_HEAD_DIM,  # head dim
                dropout_p, philox_seed, batch_philox_offset, dropout_offset,  #
                seqlen_q, seqlen_k,  # max sequence length for q and k
                start_n, start_m, num_steps,  # iteration numbers
                MASK=False,  # causal masking
                ENABLE_DROPOUT=ENABLE_DROPOUT,  # activate dropout
                USE_EXP2=USE_EXP2,
                DEBUG_TRITON=DEBUG_TRITON,
                DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
            )
        # end of GQA/MQA of dkdv
        # Write back dV and dK.
        adj_dkdv = bid * stride_dkb + hkid * stride_kh + k_start * stride_dkn
        offs_dkdv = offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk
        tl.store(DV + adj_dkdv + offs_dkdv, dv, mask=mask_kv)
        dk *= sm_scale
        tl.store(DK + adj_dkdv + offs_dkdv, dk, mask=mask_kv)

    # This part does dq
    start_m = pid * BLOCK_M2
    if start_m < seqlen_q:
        # seqlen_q > seqlen_k, no need to process these tile for dq
        if DEBUG_TRITON: print(f"end_n = start_m + BLOCK_M = {start_m} + {BLOCK_M2} = {start_m + BLOCK_M2}")  # noqa: E701
        if start_m + BLOCK_M2 < delta_qk:
            if DEBUG_TRITON: print(f"start_m + BLOCK_M2 = {start_m} + {BLOCK_M2} = {start_m + BLOCK_M2} < delta_qk of {delta_qk}")  # noqa: E701
            return

        offs_m = start_m + tl.arange(0, BLOCK_M2)
        # Mask for loading K and V
        mask_q = offs_m[:, None] < seqlen_q
        if PADDED_HEAD:
            mask_k = offs_k < ACTUAL_HEAD_DIM
            mask_q &= mask_k[None, :]
        offs_q = offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
        offs_do = offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok
        adj_kv = bid * stride_kb + hkid * stride_kh + k_start * stride_kn
        K +=  adj_kv
        V +=  adj_kv
        # If MQA / GQA, set the K and V head offsets appropriately.
        GROUP_SIZE = HQ // HK
        for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
            # seqlen_q < seqlen_k: delta_qk more kv tokens are added at the front
            #   for every M-tile
            end_n = start_m + BLOCK_M2 - delta_qk
            # clamp end_n at [0, seqlen_k]
            end_n = max(min(end_n, seqlen_k), 0)
            if DEBUG_TRITON: print(f"delta_qk: {delta_qk}; end_n: {end_n}")  # noqa: E701
            # offset input and output tensor by batch and Q/K heads
            adj_q = bid * stride_qb + hqid * stride_qh + q_start * stride_qm
            adj_do = bid * stride_dob + hqid * stride_doh + q_start * stride_dom
            adj_delta = \
                bid * stride_deltab + hqid * stride_deltah + q_start * stride_deltam
            Delta_ptr = Delta + adj_delta

            # batch_philox_offset is the ACTUALLY dropout offset
            # dropout_offset is for debug purpose and will be removed later
            batch_philox_offset = 0
            dropout_offset = 0
            if ENABLE_DROPOUT:
                batch_philox_offset = philox_offset_base + \
                                        bid * stride_dropoutb + \
                                        hqid * stride_dropouth
                dropout_offset = \
                    dropout_mask + bid * stride_dropoutb + hqid * stride_dropouth

            q = tl.load(Q + adj_q + offs_q, mask=mask_q, other=0.0)
            do = tl.load(DO + adj_do + offs_do, mask=mask_q, other=0.0)
            m = tl.load(M + adj_delta + offs_m * stride_deltam,
                        mask=offs_m < seqlen_q)
            m = m[:, None]

            MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
            # start can only be 0 at minimum
            start_n = max(end_n - BLOCK_M2, 0)
            num_steps = tl.cdiv(end_n - start_n, MASK_BLOCK_N2)
            dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
            dq = _bwd_kernel_dq(
                dq,
                q, K, V, do, m, Delta_ptr, sm_scale, #
                stride_qm, stride_qk, stride_kn,
                stride_dropoutm, stride_dropoutn,  #
                stride_deltam,
                seqlen_q, seqlen_k,  #
                BLOCK_M2, MASK_BLOCK_N2,  #
                HEAD_DIM, ACTUAL_HEAD_DIM,  #
                dropout_p, philox_seed, batch_philox_offset, dropout_offset,  #
                start_m, start_n, end_n, num_steps,  #
                MASK=True,  #
                ENABLE_DROPOUT=ENABLE_DROPOUT,
                USE_EXP2=USE_EXP2,
                DEBUG_TRITON=DEBUG_TRITON,
                DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
            )
            end_n -= num_steps * MASK_BLOCK_N2
            num_steps = tl.cdiv(end_n, BLOCK_N2)
            start_n = max(end_n - num_steps * BLOCK_N2, 0)
            if DEBUG_TRITON: print(f"unMasked: start_m: {start_m}, start_n: {start_n}, end_n: {end_n}, num_steps: {num_steps}")  # noqa: E701
            dq = _bwd_kernel_dq(
                dq,  #
                q, K, V, do, m, Delta_ptr, sm_scale, #
                stride_qm, stride_qk, stride_kn,  #
                stride_dropoutm, stride_dropoutn,  #
                stride_deltam,
                seqlen_q, seqlen_k,  #
                BLOCK_M2, BLOCK_N2,  #
                HEAD_DIM, ACTUAL_HEAD_DIM,  #
                dropout_p, philox_seed, batch_philox_offset, dropout_offset,  #
                start_m, start_n, end_n, num_steps,  #
                MASK=False,  #
                ENABLE_DROPOUT=ENABLE_DROPOUT,
                USE_EXP2=USE_EXP2,
                DEBUG_TRITON=DEBUG_TRITON,
                DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
            )
            # Write back dQ.
            adj_dq = bid * stride_dqb + hqid * stride_dqh + q_start * stride_dqm
            offs_dq = offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqk
            dq *= sm_scale
            tl.store(DQ + adj_dq + offs_dq, dq, mask=mask_q)
            # end of GQA/MQA of dq


@triton.jit
def bwd_kernel_noncausal(
    Q, K, V, sm_scale, DO, DQ, DK, DV,
    M, Delta,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_dqb, stride_dqh, stride_dqm, stride_dqk,
    stride_dkb, stride_dkh, stride_dkn, stride_dkk,
    stride_deltab, stride_deltah, stride_deltam,
    stride_dob, stride_doh, stride_dom, stride_dok,
    stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn,
    HQ, HK,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    dropout_mask, dropout_p, philox_seed, philox_offset_base,
    BLOCK_M1: tl.constexpr,  # 32
    BLOCK_N1: tl.constexpr,  # 128
    BLOCK_M2: tl.constexpr,  # 128
    BLOCK_N2: tl.constexpr,  # 32
    BLK_SLICE_FACTOR: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ACTUAL_HEAD_DIM: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_EXP2: tl.constexpr,
    DEBUG_TRITON: tl.constexpr,
    DEBUG_TRITON_DETAIL: tl.constexpr,
):
    # program ids
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    hkid = tl.program_id(2)
    # figure out varlen start and end
    q_start = 0
    k_start = 0
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k
    if IS_VARLEN:
        # Compute actual sequence lengths
        q_start = tl.load(cu_seqlens_q + bid)
        q_end = tl.load(cu_seqlens_q + bid + 1)
        k_start = tl.load(cu_seqlens_k + bid)
        k_end = tl.load(cu_seqlens_k + bid + 1)
        seqlen_q = q_end - q_start
        seqlen_k = k_end - k_start

    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    start_n = pid * BLOCK_N1

    offs_k = tl.arange(0, HEAD_DIM)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    # Mask for loading K and V
    mask_kv = offs_n[:, None] < seqlen_k
    PADDED_HEAD: tl.constexpr = (ACTUAL_HEAD_DIM != HEAD_DIM)
    if PADDED_HEAD:
        mask_k = offs_k < ACTUAL_HEAD_DIM
        mask_kv &= mask_k[None, :]
    offs_kv = offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk

    GROUP_SIZE = HQ // HK
    # K/V tensors not changed for the group
    adj_kv = bid * stride_kb + hkid * stride_kh + k_start * stride_kn
    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + adj_kv + offs_kv, mask=mask_kv, other=0.0)
    v = tl.load(V + adj_kv + offs_kv, mask=mask_kv, other=0.0)
    # If MQA / GQA, set the K and V head offsets appropriately.
    for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
        # offset input and output tensor by batch and Q/K heads
        adj_q = bid * stride_qb + hqid * stride_qh + q_start * stride_qm
        Q_ptr = Q + adj_q
        adj_do = bid * stride_dob + hqid * stride_doh + q_start * stride_dom
        DO_ptr = DO + adj_do
        adj_delta = bid * stride_deltab + hqid * stride_deltah + q_start * stride_deltam
        M_ptr = M + adj_delta
        Delta_ptr = Delta + adj_delta

        # batch_philox_offset is the ACTUALLY dropout offset
        # dropout_offset is for debug purpose and will be removed later
        batch_philox_offset = 0
        dropout_offset = 0
        if ENABLE_DROPOUT:
            batch_philox_offset = philox_offset_base + bid * stride_dropoutb + \
                                  hqid * stride_dropouth
            dropout_offset = dropout_mask + bid * stride_dropoutb + \
                             hqid * stride_dropouth

        # because there is no causal, we always start from the beginning
        start_m = 0
        num_steps = tl.cdiv(seqlen_q, BLOCK_M1)
        dk, dv = _bwd_dkdv_inner(
            dk, dv,  # output tensors
            Q_ptr, k, v, DO_ptr, M_ptr, Delta_ptr, sm_scale, # input tensors
            stride_qm, stride_qk,  # strides for q
            stride_dom, stride_dok,  # strides for o
            stride_dropoutm, stride_dropoutn,  # strides for dropout
            stride_deltam,
            BLOCK_M1, BLOCK_N1,  # block dim
            HEAD_DIM, ACTUAL_HEAD_DIM,  # head dim
            dropout_p, philox_seed, batch_philox_offset, dropout_offset,  #
            seqlen_q, seqlen_k,  # max sequence length for q and k
            start_n, start_m, num_steps,  # iteration numbers
            MASK=False,  # causal masking
            ENABLE_DROPOUT=ENABLE_DROPOUT,  # activate dropout
            USE_EXP2=USE_EXP2,
            DEBUG_TRITON=DEBUG_TRITON,
            DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
        )

    # Write back dV and dK.
    adj_dkdv = bid * stride_dkb + hkid * stride_kh + k_start * stride_dkn
    offs_dkdv = offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk
    tl.store(DV + adj_dkdv + offs_dkdv, dv, mask=mask_kv)
    dk *= sm_scale
    tl.store(DK + adj_dkdv + offs_dkdv, dk, mask=mask_kv)

    # THIS PART DOES DQ
    start_m = pid * BLOCK_M2

    offs_m = start_m + tl.arange(0, BLOCK_M2)
    # Mask for loading K and V
    mask_q = offs_m[:, None] < seqlen_q
    PADDED_HEAD: tl.constexpr = (ACTUAL_HEAD_DIM != HEAD_DIM)
    if PADDED_HEAD:
        mask_k = offs_k < ACTUAL_HEAD_DIM
        mask_q &= mask_k[None, :]
    offs_q = offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    offs_do = offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok
    adj_kv = bid * stride_kb + hkid * stride_kh + k_start * stride_kn
    K +=  adj_kv
    V +=  adj_kv
    # If MQA / GQA, set the K and V head offsets appropriately.
    GROUP_SIZE = HQ // HK
    for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
        # offset input and output tensor by batch and Q/K heads
        adj_q = bid * stride_qb + hqid * stride_qh + q_start * stride_qm
        adj_do = bid * stride_dob + hqid * stride_doh + q_start * stride_dom
        adj_delta = \
            bid * stride_deltab + hqid * stride_deltah + q_start * stride_deltam
        Delta_ptr = Delta + adj_delta

        # batch_philox_offset is the ACTUALLY dropout offset
        # dropout_offset is for debug purpose and will be removed later
        batch_philox_offset = 0
        dropout_offset = 0
        if ENABLE_DROPOUT:
            batch_philox_offset = philox_offset_base + \
                                  bid * stride_dropoutb + \
                                  hqid * stride_dropouth
            dropout_offset = \
                dropout_mask + bid * stride_dropoutb + hqid * stride_dropouth

        q = tl.load(Q + adj_q + offs_q, mask=mask_q, other=0.0)
        do = tl.load(DO + adj_do + offs_do, mask=mask_q, other=0.0)
        m = tl.load(M + adj_delta + offs_m * stride_deltam,
                    mask=offs_m < seqlen_q)
        m = m[:, None]

        # start can only be 0 at minimum
        start_n = 0
        end_n = seqlen_k
        num_steps = tl.cdiv(seqlen_k, BLOCK_N2)

        dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
        dq = _bwd_dq_inner(
            dq,  #
            q, K, V, do, m, Delta_ptr, sm_scale, #
            stride_qm, stride_qk, stride_kn,  #
            stride_dropoutm, stride_dropoutn,  #
            stride_deltam,
            seqlen_q, seqlen_k,  #
            BLOCK_M2, BLOCK_N2,  #
            HEAD_DIM, ACTUAL_HEAD_DIM,  #
            dropout_p, philox_seed, batch_philox_offset, dropout_offset,  #
            start_m, start_n, end_n, num_steps,  #
            MASK=False,  #
            ENABLE_DROPOUT=ENABLE_DROPOUT,
            USE_EXP2=USE_EXP2,
            DEBUG_TRITON=DEBUG_TRITON,
            DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
        )
        # Write back dQ.
        adj_dq = bid * stride_dqb + hqid * stride_dqh + q_start * stride_dqm
        offs_dq = offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqk
        dq *= sm_scale
        tl.store(DQ + adj_dq + offs_dq, dq, mask=mask_q)


def attention_prefill_backward_triton_split_oneKernel_impl(
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
    philox_seed,
    philox_offset,
    use_exp2: bool,
    DEBUG_TRITON: bool = False,
    DEBUG_TRITON_DETAIL: bool = False,
):
    if dq is None:
        dq = torch.empty_like(q)
    if dk is None:
        dk = torch.empty_like(k)
    if dv is None:
        dv = torch.empty_like(v)
    dq.zero_()
    dk.zero_()
    dv.zero_()

    # get strides and shape
    batch, nheads_q, nheads_k, head_size, max_seqlen_q, max_seqlen_k = \
        get_shape_from_layout(
            q, k, layout,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k
        )
    q_strides, k_strides, _, o_strides = \
        get_strides_from_layout(q, k, v, o, layout)
    stride_qb, stride_qh, stride_qm, stride_qk =  q_strides
    stride_kb, stride_kh, stride_kn, stride_kk = k_strides
    stride_ob, stride_oh, stride_om, stride_ok = o_strides
    dq_strides, dk_strides, _, do_strides = \
        get_strides_from_layout(dq, dk, dv, do, layout)
    stride_dqb, stride_dqh, stride_dqm, stride_dqk =  dq_strides
    stride_dkb, stride_dkh, stride_dkn, stride_dkk = dk_strides
    stride_dob, stride_doh, stride_dom, stride_dok = do_strides
    IS_VARLEN = layout == "thd"
    use_dropout = (dropout_p > 0.0)

    # get closest power of 2 over or equal to 32.
    padded_d_model = 1 << (head_size - 1).bit_length()
    padded_d_model = max(padded_d_model, 16)
    HEAD_DIM = padded_d_model
    ACTUAL_HEAD_DIM = head_size
    # meta-parameters
    # TODO: fix num_stages later
    NUM_WARPS, NUM_STAGES = 4, 1
    WAVES_PER_EU = 1
    PRE_BLOCK = 128
    BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
    BLK_SLICE_FACTOR = 2

    # init delta
    delta = torch.empty_like(softmax_lse)
    if IS_VARLEN:
        stride_deltab = 0
        stride_deltam, stride_deltah = delta.stride()
    else:
        stride_deltab, stride_deltah, stride_deltam = delta.stride()
    pre_grid = (triton.cdiv(max_seqlen_q, PRE_BLOCK), batch, nheads_q)
    _bwd_preprocess[pre_grid](
        o, do,
        delta,
        stride_ob, stride_oh, stride_om, stride_ok,
        stride_deltab, stride_deltah, stride_deltam,
        cu_seqlens_q, max_seqlen_q,
        BLOCK_M=PRE_BLOCK,
        HEAD_DIM=HEAD_DIM,
        ACTUAL_HEAD_DIM=ACTUAL_HEAD_DIM,
        IS_VARLEN=IS_VARLEN
    )

    # dropout mask tensor for debugging. We dump the dropout mask created in
    #   the kernel for testing
    dropout_mask = None
    stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn = \
        (0, 0 , 0 , 0)
    if use_dropout:
        dropout_mask = torch.zeros(
            (batch, nheads_q, max_seqlen_q, max_seqlen_k),
            device=q.device,
            dtype=torch.float32
        )

        if DROPOUT_USE_PYTORCH:
            if not IS_VARLEN:
                dropout_mask = create_dropout_mask(
                    dropout_p,
                    (batch, nheads_q, max_seqlen_q, max_seqlen_k),
                    seed = philox_seed
                )
            else:
                dropout_mask = create_dropout_mask_varlen(
                    dropout_p, batch, nheads_q,
                    cu_seqlens_q, cu_seqlens_k, philox_seed
                )
        stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn = \
            dropout_mask.stride()

    assert BLOCK_N1 == BLOCK_M2
    seqlen = max(max_seqlen_q, max_seqlen_k)
    grid = ((seqlen + BLOCK_N1 - 1) // BLOCK_N1, batch, nheads_k)
    assert causal, "single kernel has not been implemented for non-causal"
    if causal:
        if DEBUG_TRITON: print(f"bwd_kernel: grid = {grid}, block_size = ({BLOCK_M1, BLOCK_N1})", )  # noqa: E701
        bwd_kernel[grid](
            q, k, v, sm_scale, o, do, dq, dk, dv,
            softmax_lse, delta,
            stride_qb, stride_qh, stride_qm, stride_qk,
            stride_kb, stride_kh, stride_kn, stride_kk,
            stride_dqb, stride_dqh, stride_dqm, stride_dqk,
            stride_dkb, stride_dkh, stride_dkn, stride_dkk,
            stride_deltab, stride_deltah, stride_deltam,
            stride_dob, stride_doh, stride_dom, stride_dok,
            stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn,
            nheads_q, nheads_k,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            dropout_mask, dropout_p, philox_seed, philox_offset,
            BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2, BLK_SLICE_FACTOR,
            HEAD_DIM, ACTUAL_HEAD_DIM,
            ENABLE_DROPOUT=use_dropout,
            IS_VARLEN=IS_VARLEN,
            USE_EXP2=use_exp2,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
            waves_per_eu = WAVES_PER_EU,
            DEBUG_TRITON=DEBUG_TRITON,
            DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
        )
    else:
        bwd_kernel_noncausal[grid](
            q, k, v, sm_scale, o, do, dq, dk, dv,
            softmax_lse, delta,
            stride_qb, stride_qh, stride_qm, stride_qk,
            stride_kb, stride_kh, stride_kn, stride_kk,
            stride_deltab, stride_deltah, stride_deltam,
            stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn,
            nheads_q, nheads_k,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            dropout_mask, dropout_p, philox_seed, philox_offset,
            BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2, BLK_SLICE_FACTOR,
            HEAD_DIM, ACTUAL_HEAD_DIM,
            ENABLE_DROPOUT=use_dropout,
            IS_VARLEN=IS_VARLEN,
            USE_EXP2=use_exp2,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
            waves_per_eu = WAVES_PER_EU,
            DEBUG_TRITON=DEBUG_TRITON,
            DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
        )

    return dq, dk, dv, delta, None, None
