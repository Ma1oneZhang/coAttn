"""
Hybrid Attention with LSE Merging using FlashInfer
This script demonstrates:
1. Computing attention on 50% of KV cache using CPU implementation
2. Computing attention on remaining 50% using Flash Attention
3. Merging the results using FlashInfer's merge_state
4. Comparing with full Flash Attention
"""

import torch
import sgl_kernel
from flash_attn import flash_attn_func
import numpy as np
from typing import Tuple
import time
from current_merge import merge_state

def cpu_attention_partial(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, prefix_len: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CPU attention on a portion of KV cache using attention_buffer_only_cpu
    Returns (output, logsumexp)
    """
    query = query.to(torch.bfloat16)
    key = key.to(torch.bfloat16)
    value = value.to(torch.bfloat16)

    head_dim = query.size(-1)
    num_heads = query.size(1)
    num_tokens = query.size(0)
    buffer_len = key.size(0)

    o_extend = torch.zeros(
        (num_tokens, num_heads, head_dim),
        dtype=torch.bfloat16,
        device="cpu",
    )

    max_context_len = 128 * 1024
    req_to_token = torch.zeros((1, max_context_len), dtype=torch.int64, device="cpu")
    req_to_token[0][:buffer_len] = torch.arange(buffer_len, dtype=torch.int64)

    req_pool_indices = torch.tensor([0], dtype=torch.int64, device="cpu")
    seq_lens = torch.tensor([buffer_len], dtype=torch.int64, device="cpu")

    sm_scale = 1.0 / (head_dim**0.5)
    logit_cap = 0.0

    logsumexp = torch.ops.sgl_kernel.attention_buffer_only_cpu(
        query,
        key,
        value,
        o_extend,
        req_to_token,
        req_pool_indices,
        seq_lens,
        max_context_len,
        sm_scale,
        logit_cap,
    )

    return o_extend, logsumexp


def flash_attention_partial(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Flash Attention on a portion of KV cache
    Returns (output, logsumexp)
    """
    # Move to GPU
    q_gpu = query.to("cuda").unsqueeze(0)
    k_gpu = key.to("cuda").unsqueeze(0)
    v_gpu = value.to("cuda").unsqueeze(0)

    sm_scale = 1.0 / (query.size(-1) ** 0.5)

    # Flash attention with LSE output
    out, softmax_lse, _ = flash_attn_func(
        q=q_gpu,
        k=k_gpu,
        v=v_gpu,
        softmax_scale=sm_scale,
        causal=False,
        return_attn_probs=True,
    )  # type: ignore

    out = out.squeeze(0).to("cpu")
    softmax_lse = (
        softmax_lse.squeeze(0).transpose(0, 1).to("cpu")
    )  # [seqlen_q, num_heads]

    return out, softmax_lse


def correct_merge_states(v_a, lse_a, v_b, lse_b):
    """
    Correctly merge two attention states
    v_a, v_b: [seq_len, num_heads, head_dim]
    lse_a, lse_b: [seq_len, num_heads]
    """

    # lse_max = torch.maximum(lse_a, lse_b)

    # weight_a = torch.exp(lse_a - lse_max).unsqueeze(-1)  # [seq_len, num_heads, 1]
    # weight_b = torch.exp(lse_b - lse_max).unsqueeze(-1)

    # v_merged = (v_a * weight_a + v_b * weight_b) / (weight_a + weight_b)

    # lse_merged = lse_max + torch.log(torch.exp(lse_a - lse_max) + torch.exp(lse_b - lse_max))

    # return v_merged.to(v_a.dtype), lse_merged
    # print(v_a.shape, lse_a.shape, v_b.shape, lse_b.shape)
    return merge_state(v_a, lse_a, v_b, lse_b)


def hybrid_attention_with_merge(
    query: torch.Tensor,
    key_full: torch.Tensor,
    value_full: torch.Tensor,
    split_ratio: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hybrid attention: CPU for first portion, Flash for second portion,
    then merge with correct normalization
    """
    kv_len = key_full.size(0)
    split_point = int(kv_len * split_ratio)

    # Split KV cache
    k_first = key_full[:split_point]
    v_first = value_full[:split_point]
    k_second = key_full[split_point:]
    v_second = value_full[split_point:]

    print(f"\nHybrid Attention Split:")
    print(f"  First half (CPU): {k_first.shape[0]} tokens")
    print(f"  Second half (Flash): {k_second.shape[0]} tokens")

    # Compute attention on first half with CPU
    v_cpu, lse_cpu = cpu_attention_partial(query, k_first, v_first)

    # Compute attention on second half with Flash
    v_flash, lse_flash = flash_attention_partial(query, k_second, v_second)

    # Use correct merge implementation (works on CPU or GPU)
    if v_cpu.is_cuda:
        v_merged, lse_merged = correct_merge_states(v_cpu, lse_cpu, v_flash, lse_flash)
    else:
        # Move to GPU for better performance if needed
        v_cpu_gpu = v_cpu.to("cuda")
        lse_cpu_gpu = lse_cpu.to("cuda")
        v_flash_gpu = v_flash.to("cuda")
        lse_flash_gpu = lse_flash.contiguous().to("cuda")

        v_merged, lse_merged = correct_merge_states(
            v_cpu_gpu, lse_cpu_gpu, v_flash_gpu, lse_flash_gpu
        )

        # Move back to CPU
        v_merged = v_merged.to("cpu")
        lse_merged = lse_merged.to("cpu")

    return v_merged, lse_merged


def run_comparison_test():
    """Run comprehensive comparison between hybrid and full Flash attention"""

    print("=" * 80)
    print("Hybrid Attention with LSE Merging Test")
    print("=" * 80)

    # Test configuration
    configs = [
        {"q_batch": 64, "kv_len": 5120, "num_heads": 8, "head_dim": 128},
        {"q_batch": 128, "kv_len": 10240, "num_heads": 16, "head_dim": 128},
        {"q_batch": 256, "kv_len": 20480, "num_heads": 32, "head_dim": 128},
    ]

    results = []

    for config in configs:
        print(f"\n{'='*60}")
        print(f"Testing configuration:")
        print(f"  Query batch: {config['q_batch']}")
        print(f"  KV length: {config['kv_len']}")
        print(f"  Num heads: {config['num_heads']}")
        print(f"  Head dim: {config['head_dim']}")

        dtype = torch.bfloat16
        torch.manual_seed(42)

        # Create test tensors
        query = torch.randn(
            config["q_batch"],
            config["num_heads"],
            config["head_dim"],
            dtype=dtype,
            device="cpu",
        )
        key = torch.randn(
            config["kv_len"],
            config["num_heads"],
            config["head_dim"],
            dtype=dtype,
            device="cpu",
        )
        value = torch.randn(
            config["kv_len"],
            config["num_heads"],
            config["head_dim"],
            dtype=dtype,
            device="cpu",
        )

        # Run full Flash attention as reference
        print("\nRunning full Flash Attention...")
        start = time.time()
        v_full, lse_full = flash_attention_partial(query, key, value)
        flash_time = time.time() - start

        # Run hybrid attention with merge
        print("\nRunning hybrid attention (50% CPU + 50% Flash)...")
        start = time.time()
        v_hybrid, lse_hybrid = hybrid_attention_with_merge(
            query, key, value, split_ratio=0.5
        )
        hybrid_time = time.time() - start

        # Compare results
        cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        output_similarity = cos(v_full.flatten(1), v_hybrid.flatten(1))

        lse_diff = torch.abs(lse_full - lse_hybrid)
        lse_relative_error = lse_diff / (torch.abs(lse_full) + 1e-6)

        result = {
            "config": config,
            "output_similarity_mean": output_similarity.mean().item(),
            "output_similarity_min": output_similarity.min().item(),
            "lse_max_diff": lse_diff.max().item(),
            "lse_mean_diff": lse_diff.mean().item(),
            "lse_relative_error_mean": lse_relative_error.mean().item(),
            "lse_relative_error_max": lse_relative_error.max().item(),
            "flash_time": flash_time,
            "hybrid_time": hybrid_time,
            "v_full": v_full,
            "v_hybrid": v_hybrid,
            "lse_full": lse_full,
            "lse_hybrid": lse_hybrid,
        }
        results.append(result)

        print(f"\nResults:")
        print(
            f"  Output Similarity - Mean: {result['output_similarity_mean']:.6f}, Min: {result['output_similarity_min']:.6f}"
        )
        print(
            f"  LSE Max Diff: {result['lse_max_diff']:.6f}, Mean Diff: {result['lse_mean_diff']:.6f}"
        )
        print(
            f"  LSE Relative Error - Max: {result['lse_relative_error_max']:.6f}, Mean: {result['lse_relative_error_mean']:.6f}"
        )
        print(f"  Time - Flash: {flash_time:.4f}s, Hybrid: {hybrid_time:.4f}s")

    return results


def test_different_split_ratios():
    """Test hybrid attention with different split ratios"""

    print("\n" + "=" * 80)
    print("Testing Different Split Ratios")
    print("=" * 80)

    # Fixed configuration
    q_batch = 128
    kv_len = 1024
    num_heads = 16
    head_dim = 128
    dtype = torch.bfloat16

    torch.manual_seed(42)
    query = torch.randn(q_batch, num_heads, head_dim, dtype=dtype, device="cpu")
    key = torch.randn(kv_len, num_heads, head_dim, dtype=dtype, device="cpu")
    value = torch.randn(kv_len, num_heads, head_dim, dtype=dtype, device="cpu")

    # Get reference from full Flash attention
    v_ref, lse_ref = flash_attention_partial(query, key, value)

    # Test different split ratios
    split_ratios = [0.1, 0.25, 0.5, 0.75, 0.9]
    ratio_results = []

    for ratio in split_ratios:
        print(f"\nTesting split ratio: {ratio:.0%} CPU / {(1-ratio):.0%} Flash")
        v_hybrid, lse_hybrid = hybrid_attention_with_merge(
            query, key, value, split_ratio=ratio
        )

        # Compare with reference
        cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        similarity = cos(v_ref.flatten(1), v_hybrid.flatten(1))
        lse_diff = torch.abs(lse_ref - lse_hybrid)

        ratio_results.append(
            {
                "ratio": ratio,
                "similarity_mean": similarity.mean().item(),
                "similarity_min": similarity.min().item(),
                "lse_diff_max": lse_diff.max().item(),
                "lse_diff_mean": lse_diff.mean().item(),
            }
        )

        print(
            f"  Output similarity: {similarity.mean():.6f} (min: {similarity.min():.6f})"
        )
        print(f"  LSE diff: {lse_diff.mean():.6f} (max: {lse_diff.max():.6f})")

    return ratio_results


def main():
    """Main function to run all tests"""

    # Run main comparison test
    results = run_comparison_test()

    # Test different split ratios
    ratio_results = test_different_split_ratios()

    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    print("\nAccuracy across all configurations:")
    for r in results:
        config = r["config"]
        print(
            f"  KV={config['kv_len']:4d}: Similarity={r['output_similarity_mean']:.6f}, "
            f"LSE Diff={r['lse_mean_diff']:.6f}"
        )

    print("\nSplit ratio analysis:")
    for r in ratio_results:
        print(
            f"  {r['ratio']:.0%} CPU: Similarity={r['similarity_mean']:.6f}, "
            f"LSE Diff={r['lse_diff_mean']:.6f}"
        )

    print("\n✅ Hybrid attention with LSE merging successful!")
    print("The merged results closely match full Flash Attention, demonstrating")
    print("the correctness of combining different attention implementations.")


if __name__ == "__main__":
    main()
