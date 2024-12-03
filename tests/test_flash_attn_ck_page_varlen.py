import math

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from flash_attn import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_with_kvcache,
)

from test_flash_attn import (
    attn_bias_from_alibi_slopes,
    convert_flash_attn_S_to_softmax,
    generate_qkv,
    generate_random_padding_mask,
    _generate_block_kvcache,
    attention_ref,
    attention_kvpacked_ref,
    attention_qkvpacked_ref,
)

from flash_attn.layers.rotary import apply_rotary_emb

def is_bwd_hdim_supported(d):
    return d <= 256


def ck_randval_to_dropout_mask(randval, p):
    # If p = 0.3, randval in 255 * (0.7, 1.0] will be dropout
    # randval in 255 * [0, 0.7] will be kept
    # If return dropout_mask >=0, value will be kept
    return math.floor(255.0 * (1 - p)) - randval.to(torch.float32)


def pad_rearrange_dropout_mask_hts_to_bhss(S_dmask, cu_seqlens_q, seqlen_q_rounded, seqlen_k_rounded):
    """ pad + rearrange [nheads, total_q, max_seqlen_k] into [b, nheads, seqlen_q_rounded, seqlen_k_rounded]
    Arguments:
        S_dmask: (nheads, total_q, max_seqlen_k)
        cu_seqlens_q: (b + 1)
    Output:
        S_dmask: (b, nheads, seqlen_q_rounded, seqlen_k_rounded)
    """
    batch_size = cu_seqlens_q.numel() - 1
    seqlens_q = torch.roll(cu_seqlens_q, shifts = -1) - cu_seqlens_q
    seqlens_q = seqlens_q[0:batch_size].tolist()
    S_dmask = torch.split(S_dmask, seqlens_q, dim=1)
    # [(nheads, seqlen_q0, max_seqlen_k), (nheads, seqlen_q1, max_seqlen_k), ..., (nheads, seqlen_qb, max_seqlen_k)]
    masks = ()
    for mask in S_dmask:
        # (nheads, seqlen_qi, max_seqlen_k) -> (nheads, seqlen_q_rounded, seqlen_k_rounded)
        mask = F.pad(mask, (0, seqlen_k_rounded - mask.shape[2], 0, seqlen_q_rounded - mask.shape[1], 0, 0)).unsqueeze(1)
        masks = masks + (mask, )
    S_dmask = torch.cat(masks, dim=1)

    S_dmask = S_dmask.transpose(0, 1)
    return S_dmask


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
# @pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("local", [False, True])
# @pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("d", [32, 40, 64, 80, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize("d", [32])
@pytest.mark.parametrize("swap_sq_sk", [False, True])
# @pytest.mark.parametrize("swap_sq_sk", [False])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        # (1, 239),
        (3, 799),
        (127, 512),
        (127, 513),
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (1023, 1024),
    ],
)
# @pytest.mark.parametrize("paged_kv_block_size", [None, 256, 512])
@pytest.mark.parametrize("paged_kv_block_size", [128, 256, 512])
def test_flash_attn_varlen_causal(
    seqlen_q, seqlen_k, swap_sq_sk, d, local, paged_kv_block_size, dtype
):
    if max(seqlen_q, seqlen_k) >= 2048:
        pytest.skip()
    if swap_sq_sk:
        seqlen_q, seqlen_k = seqlen_k, seqlen_q
    device = "cuda"
    causal = True
    # set seed
    torch.random.manual_seed(0)
    batch_size = 8
    nheads = 9
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)

    if paged_kv_block_size is None:
        k = torch.randn(
            batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True
        )
        v = torch.randn(
            batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True
        )
        block_table = None
    else:
        k, v, block_table, k_cache_paged, v_cache_paged, num_blocks = _generate_block_kvcache(
            seqlen_k, paged_kv_block_size, batch_size, nheads, d, device, dtype
        )
    query_padding_mask = generate_random_padding_mask(seqlen_q, batch_size, device, mode="random")
    key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode="random")
    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)
    if paged_kv_block_size is not None:
        nblock = k_cache_paged.shape[0]

        k_cache_paged = rearrange(k_cache_paged,
                                  'nblock block_size nheads (d1 d2) -> nblock nheads d1 block_size d2',
                                  block_size=paged_kv_block_size, d1=d // 8, d2=8).contiguous()

        v_cache_paged = rearrange(v_cache_paged,
                                  'nblock block_size nheads d -> nblock nheads d block_size',
                                  block_size=paged_kv_block_size).contiguous()


    out_unpad = flash_attn_varlen_func(
        q_unpad,
        k_unpad if paged_kv_block_size is None else k_cache_paged,
        v_unpad if paged_kv_block_size is None else v_cache_paged,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        0.0,
        causal=causal,
        window_size=window_size,
        block_table=block_table,
    )
    out = output_pad_fn(out_unpad)
    out_ref, attn_ref = attention_ref(
        q,
        k,
        v,
        query_padding_mask,
        key_padding_mask,
        None,
        0.0,
        None,
        causal=causal,
        window_size=window_size,
    )
    out_pt, attn_pt = attention_ref(
        q,
        k,
        v,
        query_padding_mask,
        key_padding_mask,
        None,
        0.0,
        None,
        causal=causal,
        window_size=window_size,
        upcast=False,
        reorder_ops=True,
    )

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5
