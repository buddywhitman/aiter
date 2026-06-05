# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
batched_gemm_a8w8_smallB_blockscale — FP8 batched GEMM optimised for small B.

Fixes CU starvation for DeepSeek V4 wo_a (B=2, K=4096, N=1024, M=1..1024).

Techniques:
  1. Grid collapse + split-K: (B*M_tiles, N_tiles, split_k) grid
     → 128 WGs at B=2, M=1 → 42% CU utilisation on AMD 304-CU chip (was 5%).
  2. Per-128-block W-scales loaded as 1-D vectors — no 128× register expansion.
  3. Fused bf16 write when split_k=1 — eliminates the reduction kernel entirely.
  4. Flat reduce grid — less launch overhead than the (B*M, N_tiles) alternative.
  5. transpose_bm: accept activations in (M, B, K) ATOM natural layout — direct
     fix for the hipBLAS strided-batched contract violation in ROCm/ATOM#773.
  6. bf16_input entry-point: inline per-token-group quantization via Triton kernel
     so callers pass raw BF16 activations without a separate act_quant launch.

References: ROCm/aiter#3000, ROCm/ATOM#773 (deferred aiter fix), ROCm/ATOM#676.
"""

import os
from typing import Optional

import torch
import triton

from aiter.ops.triton._triton_kernels.gemm.batched.batched_gemm_a8w8_smallB_blockscale import (
    _batched_gemm_a8w8_smallB_blockscale_kernel,
    _split_k_reduce_flat_kernel,
    per_token_group_quant_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_FP8_E4M3_MAX = 448.0

def _default_fp8_dtype() -> torch.dtype:
    return (torch.float8_e4m3fnuz
            if os.environ.get("AITER_AMD_FP8", "1") == "1"
            else torch.float8_e5m2)

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


def per_token_group_quant(
    X: torch.Tensor,
    group_size: int = 128,
    transpose_bm: bool = False,
    fp8_dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize activations per-token per-group to fp8.

    Args:
        X: Activations. Shape (B, M, K) normally, or (M, B, K) when
           transpose_bm=True (the natural output layout of DeepSeek V4
           attention: (tokens, n_local_groups, d_per_group)).
        group_size: K-dimension quantisation group size (default 128).
        transpose_bm: If True, X is in (M, B, K) layout. The output
           X_q is always returned in canonical (B, M, K) layout.
        fp8_dtype: fp8 dtype to use. Defaults to float8_e4m3fnuz on AMD
           (AITER_AMD_FP8=1, the default) or float8_e5m2 on NVIDIA dev.

    Returns:
        Tuple of (X_q, scale):
          X_q:   (B, M, K) fp8
          scale: (B, M, K // group_size) fp32
    """
    if fp8_dtype is None:
        fp8_dtype = _default_fp8_dtype()

    if transpose_bm:
        M_dim, B_dim, K = X.shape
    else:
        B_dim, M_dim, K = X.shape

    assert K % group_size == 0
    n_groups = K // group_size

    X_q   = torch.empty(B_dim, M_dim, K, dtype=fp8_dtype, device=X.device)
    scale = torch.empty(B_dim, M_dim, n_groups, dtype=torch.float32, device=X.device)

    per_token_group_quant_kernel[(B_dim * M_dim, n_groups)](
        X, X_q, scale,
        M_dim, B_dim, K, group_size, n_groups,
        fp8_max=_FP8_E4M3_MAX,
        TRANSPOSE_BM=transpose_bm,
    )
    return X_q, scale


def batched_gemm_a8w8_smallB_blockscale_bf16(
    X: torch.Tensor,
    B_weight: torch.Tensor,
    B_scale: torch.Tensor,
    split_k: int = 8,
    BLOCK_M: int = 16,
    BLOCK_N: int = 128,
    BLOCK_K: int = 128,
    A_group_size: int = 128,
    B_block_size: int = 128,
    transpose_bm: bool = False,
    fp8_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Batched FP8 GEMM accepting raw BF16 activations (fused quantization).

    Convenience entry-point that fuses per-token-group activation quantization
    with the GEMM, so callers do not need a separate act_quant kernel launch.

    transpose_bm=True accepts X in (M, B, K) layout — the output shape of
    DeepSeek V4 attention (tokens, n_local_groups, d_per_group) — without
    requiring .transpose(0,1) from the caller. This is the direct fix for
    the hipBLAS strided-batched GEMM contract violation in ROCm/ATOM#773:
    the non-contiguous output view of torch.bmm(out=...) corrupted tiles and
    caused a GSM8K regression; this kernel avoids the view entirely.

    Args:
        X: BF16 activations. Shape (B, M, K) or (M, B, K) when transpose_bm=True.
        B_weight: FP8 weights, shape (B, N, K), pre-quantized at load time.
        B_scale: Per-block weight scales, shape (B, N, K // B_block_size), fp32.
        split_k: K split factor (8 for small M, 1 for M≥256).
        transpose_bm: Accept X in (M, B, K) ATOM natural layout.
        fp8_dtype: fp8 activation dtype. Defaults to float8_e4m3fnuz on AMD.
        out: Optional pre-allocated (B, M, N) bfloat16 output tensor.

    Returns:
        torch.Tensor: (B, M, N) bfloat16.
    """
    _LOGGER.info(
        f"BATCHED_GEMM_A8W8_SMALLB_BLOCKSCALE_BF16: X={tuple(X.shape)} "
        f"B={tuple(B_weight.shape)} transpose_bm={transpose_bm} split_k={split_k}"
    )
    A_q, A_scale = per_token_group_quant(
        X, group_size=A_group_size, transpose_bm=transpose_bm, fp8_dtype=fp8_dtype
    )
    return batched_gemm_a8w8_smallB_blockscale(
        A_q, B_weight, A_scale, B_scale,
        split_k=split_k, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        A_group_size=A_group_size, B_block_size=B_block_size, out=out,
    )
