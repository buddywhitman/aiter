# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for batched_gemm_a8w8_smallB_blockscale.

Validates correctness of the split-K FP8 batched GEMM kernel against a
dequant + einsum reference. Covers the DeepSeek V4 wo_a shapes (B=2,
K=4096, N=1024) from ROCm/aiter#3000.

Run (AMD MI300X):
    AITER_TRITON_ONLY=1 python op_tests/test_batched_gemm_smallB.py

Run (NVIDIA, software-emulated FP8, for CI):
    python op_tests/test_batched_gemm_smallB.py
"""

import argparse
import sys

import torch

from aiter.ops.triton.gemm.batched.batched_gemm_a8w8_smallB_blockscale import (
    batched_gemm_a8w8_smallB_blockscale,
)

# ---------------------------------------------------------------------------
# Quantisation helpers (self-contained, no external dependency)
# ---------------------------------------------------------------------------

# Use float8_e4m3fnuz on AMD (set AITER_AMD_FP8=1), otherwise float8_e5m2
_AMD_FP8 = __import__("os").environ.get("AITER_AMD_FP8", "0") == "1"
FP8_DTYPE = torch.float8_e4m3fnuz if _AMD_FP8 else torch.float8_e5m2
_FP8_MAX  = 448.0 if _AMD_FP8 else 57344.0


def _quantize_per_token_group(x_bf16: torch.Tensor, group_size: int = 128):
    """
    Quantise activations per token-group along K.

    Args:
        x_bf16: [B, M, K] bfloat16
        group_size: quantisation group size along K

    Returns:
        x_fp8:  [B, M, K] fp8
        scale:  [B, M, K // group_size] float32
    """
    B, M, K = x_bf16.shape
    assert K % group_size == 0
    n_groups = K // group_size
    x_grouped = x_bf16.float().reshape(B, M, n_groups, group_size)
    amax = x_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)  # [B,M,n_g,1]
    scale = (amax / _FP8_MAX).squeeze(-1)                               # [B,M,n_g]
    x_scaled = (x_grouped / amax * _FP8_MAX).clamp(-_FP8_MAX, _FP8_MAX)
    x_fp8 = x_scaled.reshape(B, M, K).to(FP8_DTYPE)
    return x_fp8, scale


def _quantize_per_block(w_bf16: torch.Tensor, block_size: int = 128):
    """
    Quantise weights per N-row block along K.

    Args:
        w_bf16: [B, N, K] bfloat16
        block_size: quantisation block size along K

    Returns:
        w_fp8:  [B, N, K] fp8
        scale:  [B, N, K // block_size] float32
    """
    B, N, K = w_bf16.shape
    assert K % block_size == 0
    n_blocks = K // block_size
    w_grouped = w_bf16.float().reshape(B, N, n_blocks, block_size)
    amax = w_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)  # [B,N,n_b,1]
    scale = (amax / _FP8_MAX).squeeze(-1)                               # [B,N,n_b]
    w_scaled = (w_grouped / amax * _FP8_MAX).clamp(-_FP8_MAX, _FP8_MAX)
    w_fp8 = w_scaled.reshape(B, N, K).to(FP8_DTYPE)
    return w_fp8, scale


def _dequant_and_matmul_ref(
    A_fp8: torch.Tensor,  # [B, M, K]
    B_fp8: torch.Tensor,  # [B, N, K]
    A_scale: torch.Tensor,  # [B, M, K // A_gs]
    B_scale: torch.Tensor,  # [B, N, K // B_bs]
    A_group_size: int = 128,
    B_block_size: int = 128,
) -> torch.Tensor:
    """Reference implementation: dequantise then einsum."""
    B_dim, M, K = A_fp8.shape
    _,     N, _ = B_fp8.shape
    n_ag = K // A_group_size
    n_bb = K // B_block_size

    A_dq = (A_fp8.float().reshape(B_dim, M, n_ag, A_group_size)
            * A_scale.unsqueeze(-1)).reshape(B_dim, M, K)
    B_dq = (B_fp8.float().reshape(B_dim, N, n_bb, B_block_size)
            * B_scale.unsqueeze(-1)).reshape(B_dim, N, K)

    return torch.einsum("bmk,bnk->bmn", A_dq, B_dq).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------

def _run_test(B, M, K, N, split_k=8, BLOCK_M=16, rel_tol=0.03, verbose=True):
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return True

    torch.manual_seed(42)
    A_bf16 = torch.randn(B, M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    W_bf16 = torch.randn(B, N, K, dtype=torch.bfloat16, device="cuda") * 0.1

    A_fp8, A_scale = _quantize_per_token_group(A_bf16, group_size=128)
    W_fp8, W_scale = _quantize_per_block(W_bf16, block_size=128)

    ref    = _dequant_and_matmul_ref(A_fp8, W_fp8, A_scale, W_scale)
    result = batched_gemm_a8w8_smallB_blockscale(
        A_fp8, W_fp8, A_scale, W_scale,
        split_k=split_k, BLOCK_M=BLOCK_M,
    )

    assert result.shape == (B, M, N), \
        f"shape mismatch: got {result.shape}, expected {(B, M, N)}"
    assert result.dtype == torch.bfloat16, \
        f"dtype mismatch: got {result.dtype}"

    abs_err = (result.float() - ref.float()).abs().max().item()
    ref_max = ref.float().abs().max().item()
    rel_err = abs_err / max(ref_max, 1e-6)

    if verbose:
        status = "PASS" if rel_err < rel_tol else "FAIL"
        print(f"  [{status}] B={B} M={M} K={K} N={N} split_k={split_k} BLOCK_M={BLOCK_M}: "
              f"rel_err={rel_err:.4%}  (tol={rel_tol:.1%})")

    assert rel_err < rel_tol, \
        (f"Relative error {rel_err:.4%} exceeds {rel_tol:.1%} threshold "
         f"(abs_err={abs_err:.5f}, ref_max={ref_max:.5f})")
    return True


# ---------------------------------------------------------------------------
# Test cases — DeepSeek V4 wo_a shapes from ROCm/aiter#3000
# ---------------------------------------------------------------------------

SHAPES = [
    # (B, M, K, N, split_k, BLOCK_M)
    (2,    1, 4096, 1024,  8, 16),   # M=1:   128 WGs on AMD MI300X → 42% CU util
    (2,    4, 4096, 1024,  8, 16),   # M=4:   128 WGs
    (2,   16, 4096, 1024,  4, 16),   # M=16:   64 WGs
    (2,   64, 4096, 1024,  8, 16),   # M=64:  512 WGs (multi-wave)
    (2,  256, 4096, 1024,  1, 32),   # M=256: fused bf16 output (no reduction kernel)
    (2, 1024, 4096, 1024,  1, 32),   # M=1024: fused bf16 output
    # Other batch sizes
    (1,    8,  512,  256,  4, 16),
    (4,    8,  512,  256,  4, 16),
]


def run_all_tests():
    print(f"\nFP8 dtype: {FP8_DTYPE} ({'AMD' if _AMD_FP8 else 'NVIDIA-compatible'})")
    print(f"Device:    {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print()

    failures = []
    for args in SHAPES:
        B, M, K, N, split_k, BLOCK_M = args
        try:
            _run_test(B, M, K, N, split_k=split_k, BLOCK_M=BLOCK_M)
        except AssertionError as e:
            failures.append(str(e))

    if failures:
        print(f"\n{len(failures)} test(s) FAILED:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print(f"\nAll {len(SHAPES)} tests PASSED.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Correctness test for batched_gemm_a8w8_smallB_blockscale."
    )
    parser.add_argument(
        "--B", type=int, default=None, help="Batch size (default: run all shapes)"
    )
    parser.add_argument("--M", type=int, default=None, help="Sequence length M")
    parser.add_argument("--K", type=int, default=4096, help="Hidden dim K")
    parser.add_argument("--N", type=int, default=1024, help="Output dim N")
    parser.add_argument("--split_k", type=int, default=8, help="split-K factor")
    parser.add_argument("--BLOCK_M", type=int, default=16, help="Tile height M")
    args = parser.parse_args()

    if args.B is not None and args.M is not None:
        _run_test(args.B, args.M, args.K, args.N,
                  split_k=args.split_k, BLOCK_M=args.BLOCK_M)
    else:
        run_all_tests()
