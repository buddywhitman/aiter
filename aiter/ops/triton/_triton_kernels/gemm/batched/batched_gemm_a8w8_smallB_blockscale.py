# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Low-level Triton kernels for batched_gemm_a8w8_smallB_blockscale.

Adds over the initial version (ROCm/aiter#3000):
  - per_token_group_quant kernel: fused BF16→FP8 activation quantization
  - transpose_bm support: accept X in (M, B, K) ATOM natural layout directly,
    avoiding the non-contiguous bmm view that caused the hipBLAS contract
    violation in ROCm/ATOM#773

References: ROCm/aiter#3000, ROCm/ATOM#773 (deferred fix), ROCm/ATOM#676.
"""

import triton
import triton.language as tl

_FP8_E4M3_MAX = 448.0


# ---------------------------------------------------------------------------
# Per-token-group activation quantization
# ---------------------------------------------------------------------------

@triton.jit
def per_token_group_quant_kernel(
    X_ptr,          # [B, M, K] or [M, B, K] bf16/fp32
    X_q_ptr,        # [B, M, K] fp8 output (always canonical layout)
    scale_ptr,      # [B, M, K // group_size] fp32 output
    M,
    B_DIM,
    K: tl.constexpr,
    group_size: tl.constexpr,
    n_groups: tl.constexpr,
    fp8_max: tl.constexpr,
    TRANSPOSE_BM: tl.constexpr,   # True → X is (M, B, K), ATOM natural layout
):
    """
    Grid: (B * M, n_groups) — one program per (token, group). No inner loop.

    TRANSPOSE_BM=True accepts activations in the (M, B, K) layout produced by
    DeepSeek V4 attention (o has shape (tokens, n_local_groups, d_per_group)).
    Avoids the caller needing .transpose(0,1) which would create a non-contiguous
    view incompatible with hipBLAS strided-batched GEMM contract (ATOM#773).
    """
    bm_id = tl.program_id(0)
    g_id  = tl.program_id(1)

    if TRANSPOSE_BM:
        b_idx = bm_id  % B_DIM
        m_idx = bm_id // B_DIM
    else:
        b_idx = bm_id // M
        m_idx = bm_id  % M

    k_start = g_id * group_size
    k_offs  = k_start + tl.arange(0, group_size)

    if TRANSPOSE_BM:
        x_ptrs = X_ptr + m_idx * B_DIM * K + b_idx * K + k_offs
    else:
        x_ptrs = X_ptr + b_idx * M * K + m_idx * K + k_offs

    x_vals  = tl.load(x_ptrs, mask=k_offs < K, other=0.0).to(tl.float32)
    abs_max = tl.maximum(tl.max(tl.abs(x_vals), axis=0), 1e-12)
    scale   = abs_max / fp8_max
    x_q     = tl.clamp(x_vals / abs_max * fp8_max, -fp8_max, fp8_max)

    # Output always in canonical [B, M, K] layout
    xq_ptrs = X_q_ptr + b_idx * M * K + m_idx * K + k_offs
    tl.store(xq_ptrs, x_q, mask=k_offs < K)
    tl.store(scale_ptr + b_idx * M * n_groups + m_idx * n_groups + g_id, scale)


# ---------------------------------------------------------------------------
# Main GEMM kernel
# ---------------------------------------------------------------------------

@triton.jit
def _batched_gemm_a8w8_smallB_blockscale_kernel(
    A_ptr,
    B_ptr,
    A_scale_ptr,
    B_scale_ptr,
    C_partial_ptr,
    C_out_ptr,
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
    FUSED_OUTPUT: tl.constexpr,
):
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

        a_ptrs = A_ptr + b_idx * M * K + m_offs[:, None] * K + k_offs[None, :]
        a_tile = tl.load(a_ptrs,
                         mask=(m_offs[:, None] < M) & (k_offs[None, :] < K),
                         other=0.0)
        a_scale_idx = (k_start + k_tile * BLOCK_K) // A_group_size
        a_scale = tl.load(
            A_scale_ptr + b_idx * M * K_DIV_A_GS + m_offs * K_DIV_A_GS + a_scale_idx,
            mask=m_offs < M, other=1.0)

        b_ptrs = B_ptr + b_idx * N * K + n_offs[:, None] * K + k_offs[None, :]
        b_tile = tl.load(b_ptrs,
                         mask=(n_offs[:, None] < N) & (k_offs[None, :] < K),
                         other=0.0)
        b_scale_idx = (k_start + k_tile * BLOCK_K) // B_block_size
        b_scale = tl.load(
            B_scale_ptr + b_idx * N * K_DIV_B_BS + n_offs * K_DIV_B_BS + b_scale_idx,
            mask=n_offs < N, other=1.0)

        a_dq = a_tile.to(tl.float32) * a_scale[:, None]
        b_dq = b_tile.to(tl.float32) * b_scale[:, None]
        acc  = tl.dot(a_dq, tl.trans(b_dq), acc, allow_tf32=True)

    m_valid = m_offs < M
    n_valid = n_offs < N

    if FUSED_OUTPUT:
        out_ptrs = (C_out_ptr
                    + b_idx * M * N + m_offs[:, None] * N + n_offs[None, :])
        tl.store(out_ptrs, acc.to(tl.bfloat16),
                 mask=m_valid[:, None] & n_valid[None, :])
    else:
        out_ptrs = (C_partial_ptr
                    + b_idx * split_k * M * N + k_split_id * M * N
                    + m_offs[:, None] * N + n_offs[None, :])
        tl.store(out_ptrs, acc.to(tl.float32),
                 mask=m_valid[:, None] & n_valid[None, :])


# ---------------------------------------------------------------------------
# Flat split-K reduction
# ---------------------------------------------------------------------------

@triton.jit
def _split_k_reduce_flat_kernel(
    C_partial_ptr,
    C_out_ptr,
    B_dim,
    M,
    N,
    split_k: tl.constexpr,
    BLOCK_FLAT: tl.constexpr,
):
    pid       = tl.program_id(0)
    flat_offs = pid * BLOCK_FLAT + tl.arange(0, BLOCK_FLAT)
    mask      = flat_offs < B_dim * M * N

    n_idx = flat_offs % N
    tmp   = flat_offs // N
    m_idx = tmp % M
    b_idx = tmp // M

    out_val = tl.zeros((BLOCK_FLAT,), dtype=tl.float32)
    for sk in range(split_k):
        ptrs = (C_partial_ptr
                + b_idx * split_k * M * N + sk * M * N + m_idx * N + n_idx)
        out_val += tl.load(ptrs, mask=mask, other=0.0)

    tl.store(C_out_ptr + b_idx * M * N + m_idx * N + n_idx,
             out_val.to(tl.bfloat16), mask=mask)
