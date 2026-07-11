"""
Latency benchmark for batched_gemm_a8w8_smallB_blockscale vs BF16 einsum,
on whatever GPU is actually present (NVIDIA here -- issue #3000's target is
AMD MI300X/MI355X, so these numbers are directional, not the number to quote
in the PR). Mirrors the DeepSeek V4 wo_a shapes from ROCm/aiter#3000.

Run: AITER_TRITON_ONLY=1 PYTHONPATH=. python op_tests/bench_batched_gemm_smallB.py
"""

import os
import time

import torch

from aiter.ops.triton.gemm.batched.batched_gemm_a8w8_smallB_blockscale import (
    batched_gemm_a8w8_smallB_blockscale,
)

_AMD_FP8 = os.environ.get("AITER_AMD_FP8", "0") == "1"
FP8_DTYPE = torch.float8_e4m3fnuz if _AMD_FP8 else torch.float8_e5m2


def _quantize_per_token_group(x_bf16: torch.Tensor, group_size: int = 128):
    B, M, K = x_bf16.shape
    x = x_bf16.view(B, M, K // group_size, group_size)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 448.0
    xq = (x / scale).clamp(-448, 448).to(FP8_DTYPE)
    return xq.view(B, M, K), scale.view(B, M, K // group_size)


def _quantize_weight(w_bf16: torch.Tensor, block: int = 128):
    B, N, K = w_bf16.shape
    w = w_bf16.view(B, N // block, block, K // block, block)
    scale = w.abs().amax(dim=(2, 4), keepdim=True).clamp(min=1e-6) / 448.0
    wq = (w / scale).clamp(-448, 448).to(FP8_DTYPE)
    return wq.view(B, N, K), scale.view(B, N // block, K // block)


def bench_one(B, M, K, N, split_k, BLOCK_M, iters=50, warmup=10):
    device = "cuda"
    x = torch.randn(B, M, K, dtype=torch.bfloat16, device=device)
    w = torch.randn(B, N, K, dtype=torch.bfloat16, device=device)
    xq, x_scale = _quantize_per_token_group(x, group_size=128)
    wq, w_scale = _quantize_weight(w, block=128)

    def run_kernel():
        return batched_gemm_a8w8_smallB_blockscale(
            xq, wq, x_scale, w_scale, split_k=split_k, BLOCK_M=BLOCK_M
        )

    def run_einsum():
        return torch.einsum("bmk,bnk->bmn", x, w)

    for fn in (run_kernel, run_einsum):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

    def timed(fn):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e6  # microseconds

    kernel_us = timed(run_kernel)
    einsum_us = timed(run_einsum)
    return kernel_us, einsum_us


if __name__ == "__main__":
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"FP8 dtype: {FP8_DTYPE} ({'AMD' if _AMD_FP8 else 'NVIDIA-emulated'})")
    print(
        "NOTE: issue #3000's target hardware is AMD MI300X/MI355X -- these "
        "numbers are directional only, not what should go in the PR.\n"
    )
    print(f"{'M':>6} {'kernel (us)':>14} {'einsum (us)':>14} {'ratio (kernel/einsum)':>22}")
    for M in (1, 16, 64, 256, 512, 1024):
        kernel_us, einsum_us = bench_one(
            B=2, M=M, K=4096, N=1024,
            split_k=8 if M <= 64 else 1,
            BLOCK_M=16 if M <= 64 else 32,
        )
        ratio = kernel_us / einsum_us
        tag = "  <- ROCm/ATOM#960 prod shape (fp8_einsum lost 1.9x to BF16 here)" if M == 512 else ""
        print(f"{M:>6} {kernel_us:>14.2f} {einsum_us:>14.2f} {ratio:>22.2f}{tag}")

    # Also try M=512 with split_k=8 explicitly (small-M dispatch), since it's
    # not obvious a priori whether the split_k=1/8 threshold at M<=64 is right
    # for this specific shape -- zufayu's #960 finding was that split-K itself
    # regressed vs no-split-K BF16 at this shape, so both dispatch choices are
    # worth comparing rather than assuming the M<=64 heuristic holds at M=512.
    print()
    kernel_us, einsum_us = bench_one(B=2, M=512, K=4096, N=1024, split_k=8, BLOCK_M=16)
    print(f"M=512, split_k=8 (forced): kernel={kernel_us:.2f}us einsum={einsum_us:.2f}us "
          f"ratio={kernel_us / einsum_us:.2f}")

    # ROCm/ATOM#960 (zufayu, 2026-06-16): explicitly NOT the B=2/TP=8 regime --
    # "gate fp8 wo_a to low-TP / large-M shapes where the einsum is actually
    # compute-bound (H>=8, M>=2048), not TP=8." These are that regime.
    print()
    print("Low-TP / large-M regime (zufayu's stated compute-bound floor, ATOM#960):")
    print(f"{'B':>4} {'M':>6} {'kernel (us)':>14} {'einsum (us)':>14} {'ratio':>8}")
    for B, M, BLOCK_M in ((8, 2048, 32), (8, 4096, 64), (16, 2048, 32)):
        kernel_us, einsum_us = bench_one(
            B=B, M=M, K=4096, N=1024, split_k=1, BLOCK_M=BLOCK_M
        )
        ratio = kernel_us / einsum_us
        print(f"{B:>4} {M:>6} {kernel_us:>14.2f} {einsum_us:>14.2f} {ratio:>8.2f}")
