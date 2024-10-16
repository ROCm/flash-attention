import torch
import math

DEBUG = False

def attention_backward_core_ref_impl(
    do, q, k, v, o, softmax_lse, sm_scale, causal, use_exp2, bwd_preprocessing_use_o
):
    # recompute attention_scores
    attention_scores = torch.matmul(q, k.transpose(-2, -1))
    if DEBUG:
        print("attention_scores:", attention_scores)

    # scale scores
    attention_scaled_scores = sm_scale * attention_scores
    if DEBUG:
        print("attention_scaled_scores:", attention_scaled_scores)

    # compute probabilities using softmax_lse
    if use_exp2:
        RCP_LN = 1 / math.log(2)
        attention_scaled_scores_base2 = attention_scaled_scores * RCP_LN
        softmax_lse_base2 = softmax_lse * RCP_LN
        p = torch.exp2(attention_scaled_scores_base2 - softmax_lse_base2.unsqueeze(-1))
    else:
        p = torch.exp(attention_scaled_scores - softmax_lse.unsqueeze(-1))

    if DEBUG:
        print("p:", p)
    # compute gradient wrt v
    dv = torch.matmul(p.transpose(-2, -1), do.to(torch.float32))

    # compute dp
    dp = torch.matmul(do, v.transpose(-2, -1))

    # calculate ds
    if bwd_preprocessing_use_o:
        delta = torch.sum(o * do, axis=-1).unsqueeze(-1).to(torch.float32)  # what OAI kernel uses
    else:
        delta = torch.sum(p * dp, axis=-1).unsqueeze(-1) # what the math says you should use
    ds = (p * (dp - delta)) * sm_scale

    # compute gradient wrt k
    dk = torch.matmul(ds.transpose(-2, -1), q.to(torch.float32))

    # compute gradient wrt q
    dq = torch.matmul(ds, k.to(torch.float32))

    # cast back to original dtype
    dq = dq.to(q.dtype)
    dk = dk.to(k.dtype)
    dv = dv.to(v.dtype)

    return dq, dk, dv, delta.squeeze(-1)

def attention_varlen_backward_pytorch_ref_impl(
    do,
    q,
    k,
    v,
    o,
    softmax_lse,
    sm_scale,
    causal,
    layout,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    use_exp2,
    bwd_preprocessing_use_o,
):
    # Ensure the layout is 'thd'
    if layout != 'thd':
        raise ValueError(f"Unsupported layout {layout}. Expected 'thd'.")

    batch_size = cu_seqlens_q.shape[0] - 1
    num_heads = q.shape[1]
    head_dim = q.shape[2]

    # Prepare lists to collect outputs
    dq_list = []
    dk_list = []
    dv_list = []
    delta_list = []

    for i in range(batch_size):
        # Get the start and end indices for the current sequence
        start_q = cu_seqlens_q[i].item()
        end_q = cu_seqlens_q[i + 1].item()
        start_k = cu_seqlens_k[i].item()
        end_k = cu_seqlens_k[i + 1].item()

        # Extract q_i, k_i, v_i, do_i, o_i, softmax_lse_i
        q_i = q[start_q:end_q, :, :]  # [L_q_i, num_heads, head_dim]
        k_i = k[start_k:end_k, :, :]  # [L_k_i, num_heads, head_dim]
        v_i = v[start_k:end_k, :, :]  # [L_k_i, num_heads, head_dim]
        do_i = do[start_q:end_q, :, :]  # [L_q_i, num_heads, head_dim]
        o_i = o[start_q:end_q, :, :]  # [L_q_i, num_heads, head_dim]
        softmax_lse_i = softmax_lse[i, :, :]  # [num_heads, L_q_i]

        # Permute to [num_heads, L_q_i, head_dim]
        q_i = q_i.permute(1, 0, 2)
        k_i = k_i.permute(1, 0, 2)
        v_i = v_i.permute(1, 0, 2)
        do_i = do_i.permute(1, 0, 2)
        o_i = o_i.permute(1, 0, 2)
        softmax_lse_i = softmax_lse_i  # Already in [num_heads, L_q_i]

        # Call the core backward function for this sequence
        dq_i, dk_i, dv_i, delta_i = attention_backward_core_ref_impl(
            do_i,
            q_i,
            k_i,
            v_i,
            o_i,
            softmax_lse_i,
            sm_scale,
            causal,
            use_exp2,
            bwd_preprocessing_use_o,
        )

        # Convert back to 'thd' layout and float16
        dq_i = dq_i.permute(1, 0, 2)  # [L_q_i, num_heads, head_dim]
        dk_i = dk_i.permute(1, 0, 2)  # [L_k_i, num_heads, head_dim]
        dv_i = dv_i.permute(1, 0, 2)  # [L_k_i, num_heads, head_dim]

        # Collect outputs
        dq_list.append(dq_i)
        dk_list.append(dk_i)
        dv_list.append(dv_i)
        delta_list.append(delta_i.unsqueeze(0))

    # Concatenate outputs
    dq = torch.cat(dq_list, dim=0)
    dk = torch.cat(dk_list, dim=0)
    dv = torch.cat(dv_list, dim=0)
    delta = torch.cat(delta_list, dim=0)  # Shape: [batch_size, num_heads, L_q_i]

    return dq, dk, dv, delta

def attention_vanilla_backward_pytorch_ref_impl(
    do,
    q,
    k,
    v,
    o,
    softmax_lse,
    sm_scale,
    causal,
    layout,
    use_exp2,
    bwd_preprocessing_use_o,
):
    if layout == "bshd":
        if DEBUG:
            print()
            print("Changing layout to bhsd!")
        do = do.transpose(1, 2).contiguous()
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        o = o.transpose(1, 2).contiguous()
    elif layout == "bhsd":
        pass
    else:
        raise ValueError(f"Unknown layout {layout}")

    dq, dk, dv, delta = attention_backward_core_ref_impl(
        do,
        q,
        k,
        v,
        o,
        softmax_lse,
        sm_scale,
        causal,
        use_exp2,
        bwd_preprocessing_use_o,
    )

    # Go back to original layout
    if layout == "bshd":
        if DEBUG:
            print()
            print("Changing back to bshd!")
        dq = dq.transpose(1, 2)
        dk = dk.transpose(1, 2)
        dv = dv.transpose(1, 2)
    elif layout == "bhsd":
        pass
    else:
        raise ValueError(f"Unknown layout {layout}")

    return dq, dk, dv, delta


def attention_backward_pytorch_ref_impl(
    do,
    q,
    k,
    v,
    o,
    softmax_lse,
    sm_scale,
    causal,
    layout,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    use_exp2,
    bwd_preprocessing_use_o,
):
    
    if layout == "thd":
        dq, dk, dv, delta = attention_varlen_backward_pytorch_ref_impl(
            do,
            q,
            k,
            v,
            o,
            softmax_lse,
            sm_scale,
            causal,
            layout,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            use_exp2,
            bwd_preprocessing_use_o,
        )
    else:
        dq, dk, dv, delta = attention_vanilla_backward_pytorch_ref_impl(
            do,
            q,
            k,
            v,
            o,
            softmax_lse,
            sm_scale,
            causal,
            layout,
            use_exp2,
            bwd_preprocessing_use_o,
        )
        

    return dq, dk, dv, delta
