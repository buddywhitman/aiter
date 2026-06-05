# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
_batched_gemm_a8w8_smallB_blockscale_kernel — low-level Triton FP8 batched GEMM
optimised for small-B (B=2, DeepSeek V4 grouped output LoRA wo_a).

Design rationale
----------------
Existing aiter FP8 batched GEMM kernels launch grid (B, M_tiles*N_tiles), which
produces only 16 work-groups for B=2, M=1, N_tiles=8 — 5% CU utilisation on a
304-CU AMD MI300X.

This kernel uses three complementary techniques:

1. Grid collapse + split-K: grid (B*M_tiles, N_tiles, split_k).
   At B=2, M=1, BLOCK_N=128, N=1024, split_k=8 → 128 WGs → 42% CU utilisation.

2. Fused bf16 output (FUSED_OUTPUT=True): when split_k==1 the main kernel writes
   bfloat16 directly, skipping the partial-sum buffer and the reduction kernel.

3. Per-128-block W-scales loaded as 1-D vectors inside the inner loop.
   Single load per (M-tile, split) avoids 128× register expansion that would
   result from broadcast of a scalar scale.

References: ROCm/aiter#3000, ROCm/ATOM#676.
"""

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Main GEMM kernel — writes partial sums OR direct bf16 output
# ---------------------------------------------------------------------------

@triton.jit
def _batched_gemm_a8w8_smallB_blockscale_kernel(
    A_ptr,
    B_ptr,
    A_scale_ptr,
    B_scale_ptr,
    C_partial_ptr,   # [B, split_k, M, N] fp32 — used only when split_k > 1
    C_out_ptr,       # [B, M, N] bf16       — used only when split_k == 1
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    split_k: tl.constexpr,
    A_group_size: tl.constexpr,
    B_block_size: tl.constexpr,
    N_A_SCALE_COLS: tl.constexpr,
    N_B_SCALE_COLS: tl.constexpr,
    K_PER_SPLIT: tl.constexpr,
    TILES_PER_SPLIT: tl.constexpr,
    K_DIV_A_GS: tl.constexpr,
    K_DIV_B_BS: tl.constexpr,
    FUSED_OUTPUT: tl.constexpr,   # True → direct bf16 write (split_k must be 1)
):
    """
    Note: this is a Triton jit function; call batched_gemm_a8w8_smallB_blockscale
    from batched_gemm_a8w8_smallB_blockscale.py, not this kernel directly.

    Grid: (B * M_tiles, N_tiles, split_k).

    Computes partial sums of C[b, m_tile, n_tile] over a K-slice of width
    K_PER_SPLIT = K // split_k and stores either:
      - fp32 partial to C_partial_ptr[b, k_split_id, m_tile, n_tile]  (split_k > 1)
      - bf16 final   to C_out_ptr   [b, m_tile, n_tile]               (split_k == 1)
    """
    batch_m_id = tl.program_id(0)
    n_tile_id  = tl.program_id(1)
    k_split_id = tl.program_id(2)

    M_tiles = tl.cdiv(M, BLOCK_M)
    b_idx   = batch_m_id // M_tiles
    m_tile  = batch_m_id  % M_tiles

    m_offs = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = n_tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    k_start = k_split_id * K_PER_SPLIT

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_tile in range(0, TILES_PER_SPLIT):
        k_offs = k_start + k_tile * BLOCK_K + tl.arange(0, BLOCK_K)

        # --- A tile: [B, M, K], row-major ---
        a_ptrs = A_ptr + b_idx * M * K + m_offs[:, None] * K + k_offs[None, :]
        a_tile = tl.load(a_ptrs,
                         mask=(m_offs[:, None] < M) & (k_offs[None, :] < K),
                         other=0.0)

        # A-scale: shape [B, M, K // A_group_size].
        # Load the 1-D slice for this (b, k_tile) — shape [BLOCK_M].
        # When BLOCK_K == A_group_size (typical: 128==128), one scale per row.
        a_scale_idx = (k_start + k_tile * BLOCK_K) // A_group_size
        a_scale = tl.load(
            A_scale_ptr + b_idx * M * K_DIV_A_GS
            + m_offs * K_DIV_A_GS + a_scale_idx,
            mask=m_offs < M, other=1.0)   # [BLOCK_M]

        # --- B tile: [B, N, K], row-major (weights stored transposed) ---
        b_ptrs = B_ptr + b_idx * N * K + n_offs[:, None] * K + k_offs[None, :]
        b_tile = tl.load(b_ptrs,
                         mask=(n_offs[:, None] < N) & (k_offs[None, :] < K),
                         other=0.0)

        # B-scale: shape [B, N, K // B_block_size].
        # Load 1-D slice for this (b, k_tile) — shape [BLOCK_N].
        b_scale_idx = (k_start + k_tile * BLOCK_K) // B_block_size
        b_scale = tl.load(
            B_scale_ptr + b_idx * N * K_DIV_B_BS
            + n_offs * K_DIV_B_BS + b_scale_idx,
            mask=n_offs < N, other=1.0)   # [BLOCK_N]

        # Dequantize: fp8 → fp32 with per-128-block scale.
        # a_scale[:, None] broadcasts over BLOCK_K columns (no 128× register copies).
        # b_scale[:, None] broadcasts over BLOCK_K columns similarly.
        a_dq = a_tile.to(tl.float32) * a_scale[:, None]   # [BLOCK_M, BLOCK_K]
        b_dq = b_tile.to(tl.float32) * b_scale[:, None]   # [BLOCK_N, BLOCK_K]
        acc  = tl.dot(a_dq, tl.trans(b_dq), acc, allow_tf32=True)

    m_valid = m_offs < M
    n_valid = n_offs < N

    if FUSED_OUTPUT:
        # split_k == 1: write directly to bf16 — no reduction kernel needed.
        out_ptrs = (C_out_ptr
                    + b_idx * M * N
                    + m_offs[:, None] * N
                    + n_offs[None, :])
        tl.store(out_ptrs, acc.to(tl.bfloat16),
                 mask=m_valid[:, None] & n_valid[None, :])
    else:
        # split_k > 1: store fp32 partial sum for later reduction.
        out_ptrs = (C_partial_ptr
                    + b_idx     * split_k * M * N
                    + k_split_id * M * N
                    + m_offs[:, None] * N
                    + n_offs[None, :])
        tl.store(out_ptrs, acc.to(tl.float32),
                 mask=m_valid[:, None] & n_valid[None, :])


# ---------------------------------------------------------------------------
# Flat reduce kernel — grid (num_output_tiles,) instead of (B*M, N_tiles).
# Reduces launch overhead for the multi-split reduction step.
# ---------------------------------------------------------------------------

@triton.jit
def _split_k_reduce_flat_kernel(
    C_partial_ptr,  # [B, split_k, M, N] fp32
    C_out_ptr,      # [B, M, N] bf16
    B_dim,
    M,
    N,
    split_k: tl.constexpr,
    BLOCK_FLAT: tl.constexpr,  # elements per program
):
    """
    Grid: (cdiv(B*M*N, BLOCK_FLAT),).
    Flat iteration over the full [B, M, N] output — one tile per program.
    Sums split_k partial fp32 results and writes bf16 output.
    """
    pid   = tl.program_id(0)
    total = B_dim * M * N

    flat_offs = pid * BLOCK_FLAT + tl.arange(0, BLOCK_FLAT)
    mask = flat_offs < total

    # Decode flat index → (b, m, n)
    n_idx = flat_offs % N
    tmp   = flat_offs // N
    m_idx = tmp % M
    b_idx = tmp // M

    # Sum split_k partials
    out_val = tl.zeros((BLOCK_FLAT,), dtype=tl.float32)
    for sk in range(split_k):
        ptrs = (C_partial_ptr
                + b_idx * split_k * M * N
                + sk    * M * N
                + m_idx * N
                + n_idx)
        out_val += tl.load(ptrs, mask=mask, other=0.0)

    out_ptrs = C_out_ptr + b_idx * M * N + m_idx * N + n_idx
    tl.store(out_ptrs, out_val.to(tl.bfloat16), mask=mask)
