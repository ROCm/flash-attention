import torch
import math
from .utils import DEBUG

def attention_forward_core_ref_impl(q, k, v, sm_scale, causal, use_exp2):
    if DEBUG:
        print()
        print("attention_forward_core_ref_impl")
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("sm_scale:", sm_scale)
        print("causal:", causal)
        print("use_exp2:", use_exp2)
    
    # Compute attention scores
    attention_scores = torch.matmul(q.to(torch.float32), k.transpose(-2, -1).to(torch.float32))
    if DEBUG:
        print("attention_scores:", attention_scores, attention_scores.shape)

    # Scale scores
    attention_scaled_scores = sm_scale * attention_scores
    if DEBUG:
        print("attention_scaled_scores:", attention_scaled_scores, attention_scaled_scores.shape)

    # Apply causal mask if necessary
    if causal:
        L_q, L_k = q.shape[1], k.shape[1]
        row_idx = torch.arange(L_q, device=q.device).unsqueeze(1)
        col_idx = torch.arange(L_k, device=q.device).unsqueeze(0)
        col_offset = L_q-L_k
        causal_mask = row_idx >= (col_offset + col_idx)
        if DEBUG:
            print("causal_mask:", causal_mask)
        # set -inf to places the causal mask is false
        attention_scaled_scores = attention_scaled_scores.masked_fill(
             torch.logical_not(causal_mask.unsqueeze(0)), float('-inf')
        )
        if DEBUG:
            print("attention_scaled_scores after causal:", attention_scaled_scores, attention_scaled_scores.shape)


    # Compute max for numerical stability
    max_scores = torch.max(attention_scaled_scores, dim=-1, keepdim=True)[0]
    if DEBUG:
        print("max_scores:", max_scores, max_scores.shape)
    if causal:
        # Replace -inf in max_scores with zeros to avoid NaN in subtraction
        max_scores = torch.where(
            torch.isinf(max_scores), torch.zeros_like(max_scores), max_scores
        )
        if DEBUG:
            print("max_scores if causal:", max_scores, max_scores.shape)

    # Shift scores
    attention_shifted_scaled_scores = attention_scaled_scores - max_scores
    if DEBUG:
            print("attention_shifted_scaled_scores:", attention_shifted_scaled_scores, attention_shifted_scaled_scores.shape)

    # Exponentiate
    if use_exp2:
        RCP_LN = 1 / math.log(2)
        exp_scores = torch.exp2(RCP_LN * attention_shifted_scaled_scores)
    else:
        exp_scores = torch.exp(attention_shifted_scaled_scores)

    if DEBUG:
        print("exp_scores:", exp_scores, exp_scores.shape)

    # Sum of exponentials
    sum_exp_scores = torch.sum(exp_scores, dim=-1, keepdim=True)
    if DEBUG:
        print("sum_exp_scores:", sum_exp_scores, sum_exp_scores.shape)
    if causal:
        # if sum of exp scores is 0.0 it means scores where -inf, we cannot compute softmax and softmax_lse. Setting to 1 deals with -inf case cleanly 
        sum_exp_scores = torch.where(
        sum_exp_scores == 0,
        torch.ones_like(sum_exp_scores),
        sum_exp_scores
        )
    if DEBUG:
        print("sum_exp_scores:", sum_exp_scores, sum_exp_scores.shape)

    # Compute softmax probabilities
    softmax = exp_scores / sum_exp_scores

    if DEBUG:
        print("softmax:", softmax, softmax.shape)

    # Compute log-sum-exp
    if use_exp2:
        LN2 = math.log(2)
        RCP_LN = 1 / math.log(2)
        max_scores_base2 = max_scores * RCP_LN
        softmax_lse_base2 = max_scores_base2 + torch.log2(sum_exp_scores)
        softmax_lse = softmax_lse_base2 * LN2
        softmax_lse.squeeze_(-1)
    else:
        softmax_lse = max_scores + torch.log(sum_exp_scores)
        softmax_lse = softmax_lse.squeeze(-1)

    if DEBUG:
        print("softmax_lse:", softmax_lse, softmax_lse.shape)

    # Compute output
    o = torch.matmul(softmax, v.to(torch.float32)).to(torch.float16)
    if DEBUG:
        print("o:", o, o.shape)

    return o, softmax_lse, exp_scores, softmax, attention_shifted_scaled_scores, attention_scaled_scores, attention_scores

def attention_vanilla_forward_pytorch_ref_impl(q, k, v, sm_scale, causal, layout, use_exp2):
    """Compute reference output and softmax_lse using PyTorch's built-in function"""

    # Ensure the layout is 'bhsd'
    if layout == "bshd":
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
    elif layout != "bhsd":
        raise ValueError(f"Unknown layout {layout}")

    # Prepare tensors
    batch_size, nheads_q, seq_len_q, head_dim = q.shape
    batch_size, nheads_k, seq_len_k, head_dim = k.shape
    group_size = nheads_q // nheads_k
    if nheads_q % nheads_k != 0:
        raise ValueError("nheads_q must be divisible by nheads_k")

    if group_size != 1:
        # MQA or GQA case
        # Reshape q to [batch_size, nheads_k, group_size, seq_len_q, head_dim]
        q = q.reshape(batch_size, nheads_k, group_size, seq_len_q, head_dim)
        # Expand k and v to match group_size
        k = k.unsqueeze(2).expand(-1, -1, group_size, -1, -1)
        v = v.unsqueeze(2).expand(-1, -1, group_size, -1, -1)
        # Flatten the first three dimensions for computation
        q = q.reshape(batch_size * nheads_k * group_size, seq_len_q, head_dim)
        k = k.reshape(batch_size * nheads_k * group_size, seq_len_k, head_dim)
        v = v.reshape(batch_size * nheads_k * group_size, seq_len_k, head_dim)
    else:
        q = q.reshape(batch_size * nheads_q, seq_len_q, head_dim)
        k = k.reshape(batch_size * nheads_k, seq_len_k, head_dim)
        v = v.reshape(batch_size * nheads_k, seq_len_k, head_dim)

    # Call the core attention function
    o, softmax_lse, exp_scores, softmax, attention_shifted_scaled_scores, attention_scaled_scores, attention_scores = attention_forward_core_ref_impl(
        q, k, v, sm_scale, causal, use_exp2
    )

    if group_size != 1:
        # Reshape outputs back to original dimensions
        o = o.reshape(batch_size, nheads_k, group_size, seq_len_q, head_dim)
        o = o.reshape(batch_size, nheads_q, seq_len_q, head_dim)
        softmax_lse = softmax_lse.reshape(batch_size, nheads_k, group_size, seq_len_q)
        softmax_lse = softmax_lse.reshape(batch_size, nheads_q, seq_len_q)
        exp_scores = exp_scores.reshape(batch_size, nheads_k, group_size, seq_len_q, seq_len_k)
        exp_scores = exp_scores.reshape(batch_size, nheads_q, seq_len_q, seq_len_k)
        softmax = softmax.reshape(batch_size, nheads_k, group_size, seq_len_q, seq_len_k)
        softmax = softmax.reshape(batch_size, nheads_q, seq_len_q, seq_len_k)
        attention_scaled_scores = attention_scaled_scores.reshape(batch_size, nheads_k, group_size, seq_len_q, seq_len_k)
        attention_scaled_scores = attention_scaled_scores.reshape(batch_size, nheads_q, seq_len_q, seq_len_k)
    else:
        # Standard case
        o = o.reshape(batch_size, nheads_q, seq_len_q, head_dim)
        softmax_lse = softmax_lse.reshape(batch_size, nheads_q, seq_len_q)
        exp_scores = exp_scores.reshape(batch_size, nheads_q, seq_len_q, seq_len_k)
        softmax = softmax.reshape(batch_size, nheads_q, seq_len_q, seq_len_k)
        attention_shifted_scaled_scores = attention_shifted_scaled_scores.reshape(batch_size, nheads_q, seq_len_q, seq_len_k)
        attention_scaled_scores = attention_scaled_scores.reshape(batch_size, nheads_q, seq_len_q, seq_len_k)
        attention_scores = attention_scores.reshape(batch_size, nheads_q, seq_len_q, seq_len_k)

    # Restore original layout if necessary
    if layout == "bshd":
        o = o.transpose(1, 2)

    return o, softmax_lse, exp_scores, softmax, attention_shifted_scaled_scores, attention_scaled_scores, attention_scores


def attention_varlen_forward_pytorch_ref_impl(
    q,
    k,
    v,
    sm_scale,
    causal,
    layout,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    use_exp2
):
    # Ensure the layout is 'thd'
    if layout != 'thd':
        raise ValueError(f"Unsupported layout {layout}. Expected 'thd'.")

    batch_size = cu_seqlens_q.shape[0] - 1
    nheads_q, nheads_k = q.shape[1], k.shape[1]
    head_dim = q.shape[2]

    # Pre-allocate outputs
    total_L_q = q.shape[0]

    o = torch.empty((total_L_q, nheads_q, head_dim), dtype=q.dtype, device=q.device)
    softmax_lse = torch.empty((total_L_q, nheads_q), dtype=torch.float32, device=q.device)

    # Compute group_size for MQA/GQA handling
    group_size = nheads_q // nheads_k
    if nheads_q % nheads_k != 0:
        raise ValueError("nheads_q must be divisible by nheads_k")

    for i in range(batch_size):
        # Get the start and end indices for the current sequence
        start_q = cu_seqlens_q[i].item()
        end_q = cu_seqlens_q[i + 1].item()
        start_k = cu_seqlens_k[i].item()
        end_k = cu_seqlens_k[i + 1].item()

        seqlen_q = end_q - start_q
        seqlen_k = end_k - start_k

        if DEBUG:
            print(f"Batch {i} with seqlen_q = {seqlen_q}, seqlen_k = {seqlen_k}, Hq= {nheads_q}, Hk = {nheads_k}")

        # Extract q_i, k_i, v_i
        q_i = q[start_q:end_q, :, :]  # [L_q_i, nheads_q, head_dim]
        k_i = k[start_k:end_k, :, :]  # [L_k_i, nheads_k, head_dim]
        v_i = v[start_k:end_k, :, :]  # [L_k_i, nheads_k, head_dim]

        # Permute to [nheads, L_q_i, head_dim]
        q_i = q_i.permute(1, 0, 2)
        k_i = k_i.permute(1, 0, 2)
        v_i = v_i.permute(1, 0, 2)

        # Handle MQA/GQA by adjusting shapes based on group_size
        if group_size != 1:
            # Reshape q_i to [nheads_k, group_size, L_q_i, head_dim]
            q_i = q_i.reshape(nheads_k, group_size, seqlen_q, head_dim)
            # Expand k_i and v_i to match group_size
            k_i = k_i.unsqueeze(1).expand(-1, group_size, -1, -1)
            v_i = v_i.unsqueeze(1).expand(-1, group_size, -1, -1)
            # Flatten the first two dimensions for computation
            q_i = q_i.reshape(nheads_k * group_size, seqlen_q, head_dim)
            k_i = k_i.reshape(nheads_k * group_size, seqlen_k, head_dim)
            v_i = v_i.reshape(nheads_k * group_size, seqlen_k, head_dim)
        else:
            # Standard case
            q_i = q_i.reshape(nheads_q, seqlen_q, head_dim)
            k_i = k_i.reshape(nheads_k, seqlen_k, head_dim)
            v_i = v_i.reshape(nheads_k, seqlen_k, head_dim)

        # Call the core attention function for this sequence
        (
            o_i,
            softmax_lse_i,
            exp_scores_i,
            softmax_i,
            attention_shifted_scaled_scores_i,
            attention_scaled_scores_i,
            attention_scores_i,
        ) = attention_forward_core_ref_impl(q_i, k_i, v_i, sm_scale, causal, use_exp2)

        # Reshape outputs back to original dimensions
        if group_size != 1:
            # Reshape outputs to [nheads_k, group_size, seqlen_q, head_dim]
            o_i = o_i.reshape(nheads_k, group_size, seqlen_q, head_dim)
            # Combine the first two dimensions back to nheads_q
            o_i = o_i.reshape(nheads_q, seqlen_q, head_dim)
            # Reshape softmax_lse_i similarly
            softmax_lse_i = softmax_lse_i.reshape(nheads_k, group_size, seqlen_q)
            softmax_lse_i = softmax_lse_i.reshape(nheads_q, seqlen_q)
        else:
            # Outputs are already in the correct shape
            pass

        # Convert back to 'thd' layout and float16
        o_i = o_i.permute(1, 0, 2).to(torch.float16)  # [L_q_i, nheads_q, head_dim]
        softmax_lse_i = softmax_lse_i.permute(1, 0)  # [L_q_i, nheads_q]

        # Place outputs in pre-allocated tensors
        o[start_q:end_q, :, :] = o_i
        softmax_lse[start_q:end_q, :] = softmax_lse_i

    return (
        o,
        softmax_lse,
        None,
        None,
        None,
        None,
        None,
    )



def attention_forward_pytorch_ref_impl(
    q,
    k,
    v,
    sm_scale,
    causal,
    layout,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    use_exp2
    ):
    if DEBUG:
        print()
        print("attention_forward_pytorch_ref_impl")
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("sm_scale:", sm_scale)
        print("causal:", causal)
        print("layout:", layout)
        print("cu_seqlens_q:", cu_seqlens_q)
        print("cu_seqlens_k:", cu_seqlens_k)
        print("max_seqlen_q:", max_seqlen_q)
        print("max_seqlen_k:", max_seqlen_k)
        print("use_exp2:", use_exp2)

     # compute reference
    if layout == "thd":
        (
            o_ref,
            softmax_lse_ref,
            exp_scores_ref,
            softmax_ref,
            attention_shifted_scaled_scores_ref,
            attention_scaled_scores_ref,
            attention_scores_ref,
        ) = attention_varlen_forward_pytorch_ref_impl(
            q.clone(), 
            k.clone(), 
            v.clone(), 
            sm_scale, 
            causal, 
            layout,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            use_exp2,
        )
    else:
        (
            o_ref,
            softmax_lse_ref,
            exp_scores_ref,
            softmax_ref,
            attention_shifted_scaled_scores_ref,
            attention_scaled_scores_ref,
            attention_scores_ref,
        ) = attention_vanilla_forward_pytorch_ref_impl(
            q.clone(), k.clone(), v.clone(), sm_scale, causal, layout, use_exp2
        )

    if DEBUG:
        print()
        print("attention_forward_pytorch_ref_impl outputs")
        print("o_ref:", o_ref, o_ref.shape)
        print("softmax_lse_ref:", softmax_lse_ref, softmax_lse_ref.shape)
        print("exp_scores_ref:", exp_scores_ref, exp_scores_ref.shape if exp_scores_ref is not None else None)

    return (
            o_ref,
            softmax_lse_ref,
            exp_scores_ref,
            softmax_ref,
            attention_shifted_scaled_scores_ref,
            attention_scaled_scores_ref,
            attention_scores_ref,
    )


def compute_alibi_tensor_ref(alibi_slopes, seqlen_q, seqlen_k):
    q_idx = torch.arange(seqlen_q, dtype=torch.int32, device="cuda").unsqueeze(-1)  # (N_CTX_Q, 1)
    k_idx = torch.arange(seqlen_k, dtype=torch.int32, device="cuda").unsqueeze(0)  # (1, N_CTX_K)
    relative_pos = torch.abs(q_idx + seqlen_k - seqlen_q - k_idx)  # (N_CTX_Q, N_CTX_K)
    return -1 * alibi_slopes.unsqueeze(-1).unsqueeze(-1) * relative_pos  # (Z, H, N_CTX_Q, N_CTX_K)