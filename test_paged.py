import torch
import sgl_kernel
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

torch.random.manual_seed(0)


def cdiv(a, b):
    return (a + b - 1) // b


block_size = 16
head_dim = 128
q_head = 28
kv_head = 4
seq_lens = [1025, 1000, 10241, 9999, 101]
q_lens = [1024, 1, 124, 1231, 1]
total_q_len = sum(q_lens)

kv_block_cnt = int(cdiv(torch.max(torch.tensor(seq_lens)).item(), block_size))

assert len(seq_lens) == len(q_lens)


def generate_data():
    global q_lens, seq_lens
    q = torch.randn((total_q_len, q_head, head_dim), device="cpu", dtype=torch.bfloat16)
    k = torch.randn(
        (kv_block_cnt, block_size, kv_head, head_dim),
        device="cpu",
        dtype=torch.bfloat16,
    )
    v = torch.randn(
        (kv_block_cnt, block_size, kv_head, head_dim),
        device="cpu",
        dtype=torch.bfloat16,
    )
    block_table = torch.full(
        (len(seq_lens), kv_block_cnt), -1, device="cpu", dtype=torch.int32
    )
    for i, seqlen in enumerate(seq_lens):
        block_cnt = int(cdiv(seqlen, block_size))
        block_table[i, :block_cnt] = torch.arange(block_cnt, dtype=torch.int32)
    q_lens = torch.tensor(q_lens, device="cpu", dtype=torch.int32)
    cu_q_lens = torch.cat(
        [torch.zeros(1, device="cpu", dtype=torch.int32), q_lens.cumsum(dim=0)]
    ).to(torch.int32)
    seq_lens = torch.tensor(seq_lens, device="cpu", dtype=torch.int32)
    return (
        q,
        k,
        v,
        block_table,
        q_lens,
        cu_q_lens,
        seq_lens,
    )


def test_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_table: torch.Tensor,
    q_lens: torch.Tensor,
    cu_q_lens: torch.Tensor,
    seq_lens: torch.Tensor,
):
    q = q.cuda()
    k = k.cuda()
    v = v.cuda()
    block_table = block_table.cuda()
    cu_q_lens = cu_q_lens.cuda()
    seq_lens = seq_lens.cuda()
    q_lens = q_lens.cuda()

    max_seqlen_q = int(q_lens.max().item())
    max_seqlen_k = int(seq_lens.max().item())

    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        max_seqlen_q,
        cu_seqlens_q=cu_q_lens,
        max_seqlen_k=max_seqlen_k,
        block_table=block_table,
        seqused_k=seq_lens,
        causal=False,
        dropout_p=0.0,
        return_softmax_lse=True,
    )
    return out, lse


def test_sgl_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_table: torch.Tensor,
    q_lens: torch.Tensor,
    cu_q_lens: torch.Tensor,
    seq_lens: torch.Tensor,
):
    output = torch.zeros(
        (total_q_len, q_head, head_dim), device="cpu", dtype=torch.bfloat16
    )
    lse = torch.zeros(q.shape[:2], device="cpu", dtype=torch.float32)
    sm_scale = 1.0 / (head_dim**0.5)
    logit_cap = 0.0

    torch.ops.sgl_kernel.flash_paged_attention_cpu(
        query=q,
        k_buffer=k,
        v_buffer=v,
        output=output,
        lse=lse,
        block_table=block_table,
        cu_q_lens=cu_q_lens,
        seq_lens=seq_lens,
        q_lens=q_lens,
        sm_scale=sm_scale,
        logit_cap=logit_cap,
    )
    return output, lse


if __name__ == "__main__":
    input()
    torch.set_printoptions(sci_mode=True, linewidth=200)
    q, k, v, block_table, q_lens, cu_q_lens, seq_lens = generate_data()

    print("=" * 80)
    print("Test: flase_paged_attention_cpu")
    print("=" * 80)
    output_fa, lse_fa = test_flash_attn(
        q, k, v, block_table, q_lens, cu_q_lens, seq_lens
    )
    # print(output_fa.shape)
    # print(output_fa)
    # exit(0)
    # print(lse_fa)
    output_sgl, lse_sgl = test_sgl_kernel(
        q, k, v, block_table, q_lens, cu_q_lens, seq_lens
    )
    # lse_fa = lse_fa.reshape(-1, q_head)
    # print(lse_fa.shape)
    # print(lse_fa)
    lse_fa = lse_fa.transpose(0, 1)
    # print(lse_sgl[:, 0])
    print("Output max difference:", torch.max(torch.abs(output_fa.cpu() - output_sgl)))
    print(
        "Output mean difference:", torch.mean(torch.abs(output_fa.cpu() - output_sgl))
    )
    print("LSE max difference:", torch.max(torch.abs(lse_fa.cpu() - lse_sgl)))
    print("LSE mean difference:", torch.mean(torch.abs(lse_fa.cpu() - lse_sgl)))

    print("\n" + "=" * 80)
    # print("Test 2: decode_attention_cpu")
    # print("=" * 80)
    # # output_decode, lse_decode = test_decode_attention_cpu(q, k, v)
    # output_fa_reshaped = output_fa.cpu().reshape(-1, q_head, head_dim)

    # # Output comparison
    # output_max_diff = torch.max(torch.abs(output_fa_reshaped - output_decode))
    # output_mean_diff = torch.mean(torch.abs(output_fa_reshaped - output_decode))

    # # LSE comparison
    # lse_max_diff = torch.max(torch.abs(lse_fa.cpu() - lse_decode))
    # lse_mean_diff = torch.mean(torch.abs(lse_fa.cpu() - lse_decode))

    # print(f"Output comparison:")
    # print(f"  Max absolute difference: {output_max_diff:.6e}")
    # print(f"  Mean absolute difference: {output_mean_diff:.6e}")
    # print(f"\nLSE comparison:")
    # print(f"  Max absolute difference: {lse_max_diff:.6e}")
    # print(f"  Mean absolute difference: {lse_mean_diff:.6e}")

    # # Detailed statistics
    # print(f"\nDetailed statistics:")
    # print(f"  Output range: [{output_decode.min():.4f}, {output_decode.max():.4f}]")
    # print(f"  LSE range: [{lse_decode.min():.4f}, {lse_decode.max():.4f}]")
    # print(
    #     f"  Flash attn output range: [{output_fa_reshaped.min():.4f}, {output_fa_reshaped.max():.4f}]"
    # )
    print(f"  Flash attn LSE range: [{lse_fa.min():.4f}, {lse_fa.max():.4f}]")

    print("=" * 80)
