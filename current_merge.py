import torch
import triton
import triton.language as tl


@triton.jit
def state_normalize(o, m, d):
    o = o / d
    return o, m, d


@triton.jit
def state_merge(o, m, d, other_o, other_m, other_d):
    m_max = tl.maximum(m, other_m)
    d = d * tl.exp(m - m_max) + other_d * tl.exp(other_m - m_max)
    o = o * tl.exp(m - m_max) + other_o * tl.exp(other_m - m_max)
    return o, m_max, d


@triton.jit
def merge_state_kernel(
    v_a_ptr,
    s_a_ptr,
    v_b_ptr,
    s_b_ptr,
    v_merged_ptr,
    s_merged_ptr,
    num_heads,
    head_dim,
    bdx: tl.constexpr,
    bdy: tl.constexpr,
):
    pos = tl.program_id(axis=0)
    for tx in tl.range(bdx):
        for head_idx in tl.range(bdy):
            s_a_val = tl.load(s_a_ptr + pos * num_heads + head_idx)
            s_b_val = tl.load(s_b_ptr + pos * num_heads + head_idx)

            offsets = (pos * num_heads + head_idx) * head_dim + tx
            v_a = tl.load(v_a_ptr + offsets)
            v_b = tl.load(v_b_ptr + offsets)

            v_merged, s_max, d = state_merge(
                o=v_a, m=s_a_val, d=1, other_o=v_b, other_m=s_b_val, other_d=1
            )
            v_merged, s_max, d = state_normalize(v_merged, s_max, d)
            v_merged_offset = (pos * num_heads + head_idx) * head_dim + tx
            tl.store(v_merged_ptr + v_merged_offset, v_merged)

            if s_merged_ptr:
                tl.store(
                    s_merged_ptr + pos * num_heads + head_idx,
                    tl.log(d) + s_max,
                )


def merge_state(
    v_a: torch.Tensor, s_a: torch.Tensor, v_b: torch.Tensor, s_b: torch.Tensor
):
    assert v_a.size(0) == s_a.size(0)
    assert v_a.size(1) == s_b.size(1)
    s_a = s_a.to(torch.float32)
    s_b = s_b.to(torch.float32)
    seq_len = v_a.size(0)
    num_heads = v_a.size(1)
    head_dim = v_a.size(2)
    v_merged = torch.empty_like(v_a).to(s_a.device)
    s_merged = torch.empty((seq_len, num_heads)).to(s_a.device)
    bdx = head_dim
    bdy = num_heads

    merge_state_kernel[lambda meta: (seq_len,)](
        v_a, s_a, v_b, s_b, v_merged, s_merged, num_heads, head_dim, bdx=bdx, bdy=bdy
    )

    return v_merged, s_merged
