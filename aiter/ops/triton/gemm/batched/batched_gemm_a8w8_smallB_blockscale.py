# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
batched_gemm_a8w8_smallB_blockscale — FP8 batched GEMM optimised for small B.

Fixes CU starvation for DeepSeek V4 wo_a (B=2, K=4096, N=1024, M=1..1024).

Techniques:
  1. Grid collapse: B folded into M → (B*M_tiles, N_tiles, split_k) grid.
  2. Split-K: K split across work-groups → 128 WGs at B=2, M=1 (42% AMD CU util).
  3. Per-128-block W-scales in kernel (blockscale, no dequant/requant precision loss).
  4. Fused bf16 write when split_k=1 — skips the partial-sum buffer entirely.
  5. Flat reduce grid — reduces launch overhead for the multi-split reduction.

References: ROCm/aiter#3000, ROCm/ATOM#676.
"""

from typing import Optional
import torch
import triton

from aiter.ops.triton._triton_kernels.gemm.batched.batched_gemm_a8w8_smallB_blockscale import (
    _batched_gemm_a8w8_smallB_blockscale_kernel,
    _split_k_reduce_flat_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def batched_gemm_a8w8_smallB_blockscale(
    A: torch.Tensor,           # [B, M, K] fp8 (float8_e4m3fnuz on AMD)
    B_weight: torch.Tensor,    # [B, N, K] fp8, weights stored row-major (N×K)
    A_scale: torch.Tensor,     # [B, M, K // A_group_size] fp32
    B_scale: torch.Tensor,     # [B, N, K // B_block_size] fp32
    split_k: int = 8,
    BLOCK_M: int = 16,
    BLOCK_N: int = 128,
    BLOCK_K: int = 128,
    A_group_size: int = 128,
    B_block_size: int = 128,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Batched FP8 GEMM for small B (B=2, DeepSeek V4 wo_a).

    Computes C[b] = A[b] @ B_weight[b].T for b in range(B).

    Args:
        A (torch.Tensor): Activations, shape (B, M, K), fp8.
        B_weight (torch.Tensor): Weights, shape (B, N, K), fp8.
            Stored as (N, K) per batch — the kernel transposes internally.
        A_scale (torch.Tensor): Per-token-group activation scales,
            shape (B, M, K // A_group_size), fp32.
        B_scale (torch.Tensor): Per-block weight scales,
            shape (B, N, K // B_block_size), fp32.
        split_k (int): K-dimension split factor. Larger values increase
            parallelism for small M (recommended: 8 for M≤16, 1 for M≥256).
        BLOCK_M (int): Tile height in the M dimension.
        BLOCK_N (int): Tile width in the N dimension (default 128).
        BLOCK_K (int): Inner loop tile size (must equal A_group_size and
            B_block_size in the typical 128-block-scale regime).
        A_group_size (int): Quantisation group size along K for activations.
        B_block_size (int): Quantisation block size along K for weights.
        out (Optional[torch.Tensor]): Pre-allocated output of shape (B, M, N),
            dtype bfloat16. If None, a new tensor is allocated.

    Returns:
        torch.Tensor: Output of shape (B, M, N), bfloat16.
    """
    _LOGGER.info(
        f"BATCHED_GEMM_A8W8_SMALLB_BLOCKSCALE: A={tuple(A.shape)} "
        f"B={tuple(B_weight.shape)} A_scale={tuple(A_scale.shape)} "
        f"B_scale={tuple(B_scale.shape)} split_k={split_k}"
    )

    A       = A.contiguous()
    B_weight = B_weight.contiguous()

    B_dim, M, K = A.shape
    _,     N, _ = B_weight.shape

    assert K % BLOCK_K          == 0, f"K={K} must be divisible by BLOCK_K={BLOCK_K}"
    assert K % split_k          == 0, f"K={K} must be divisible by split_k={split_k}"
    assert BLOCK_K % A_group_size == 0, \
        f"BLOCK_K={BLOCK_K} must be divisible by A_group_size={A_group_size}"
    assert BLOCK_K % B_block_size == 0, \
        f"BLOCK_K={BLOCK_K} must be divisible by B_block_size={B_block_size}"

    K_PER_SPLIT     = K // split_k
    TILES_PER_SPLIT = K_PER_SPLIT // BLOCK_K
    K_DIV_A_GS      = K // A_group_size
    K_DIV_B_BS      = K // B_block_size

    if out is None:
        C_out = torch.empty(B_dim, M, N, dtype=torch.bfloat16, device=A.device)
    else:
        assert out.shape == (B_dim, M, N), \
            f"pre-allocated out has wrong shape {out.shape}, expected {(B_dim, M, N)}"
        assert out.dtype == torch.bfloat16, \
            f"pre-allocated out must be bfloat16, got {out.dtype}"
        C_out = out

    M_tiles   = triton.cdiv(M, BLOCK_M)
    grid_main = (B_dim * M_tiles, triton.cdiv(N, BLOCK_N), split_k)

    fused = (split_k == 1)

    # Allocate partial buffer only when split_k > 1.
    # When fused=True, C_partial_ptr is unused — pass a dummy (C_out) to keep
    # the kernel signature valid without wasting memory.
    C_partial = (
        torch.empty(B_dim, split_k, M, N, dtype=torch.float32, device=A.device)
        if not fused
        else C_out   # dummy; kernel will not write to it when FUSED_OUTPUT=True
    )

    _batched_gemm_a8w8_smallB_blockscale_kernel[grid_main](
        A, B_weight, A_scale, B_scale,
        C_partial, C_out,
        M, N, K,
        BLOCK_M, BLOCK_N, BLOCK_K,
        split_k, A_group_size, B_block_size,
        1, 1,                          # N_A/B_SCALE_COLS: kept for Triton signature
        K_PER_SPLIT, TILES_PER_SPLIT,
        K_DIV_A_GS, K_DIV_B_BS,
        FUSED_OUTPUT=fused,
        num_stages=1,
    )

    if not fused:
        # Flat reduce: fewer programs than (B*M, N_tiles), less launch overhead.
        BLOCK_FLAT = 512
        grid_reduce = (triton.cdiv(B_dim * M * N, BLOCK_FLAT),)
        _split_k_reduce_flat_kernel[grid_reduce](
            C_partial, C_out,
            B_dim, M, N, split_k, BLOCK_FLAT,
        )

    return C_out
