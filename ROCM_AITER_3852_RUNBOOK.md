# ROCm/aiter#3852 — gfx942/cu304 tuned config gap — runbook for AMD hardware

## What this is

External customer (mangoboost.io) reported 6 DeepSeek-R1-style TP8 vLLM shapes
falling back to the untuned default A8W8 blockscale GEMM config on MI300X
(gfx942/cu304): repeated `not found tuned config in a8w8_blockscale_tuned_gemm.csv`
warnings for 3 prefill shapes (M=32768) and 3 decode shapes (M=2048).

Verified before touching any code: gfx942/cu304 has only 182 tuned rows total
vs gfx950/cu256's 20945 — a real, large coverage gap, not specific to these
6 shapes. Zero maintainer engagement on the issue as of 2026-07-11.

## What's already done (this session, no AMD hardware needed)

- `aiter/configs/a8w8_blockscale_untuned_gemm.csv`: the 3 decode shapes
  (M=2048, N=4608/K=7168, N=7168/K=2304, N=7168/K=256) **already existed** in
  this file (lines 55-57) — they're queued for tuning, just never actually run
  on gfx942 hardware. The 3 prefill shapes (M=32768) did **not** exist
  anywhere (max M in the file was 20480) — added as 3 new rows at the end.
- Confirmed via `python3 -c "import csv; ..."` the file is still well-formed
  (86 data rows, all 3-field, header intact).

## What needs real gfx942/cu304 hardware tomorrow

```bash
cd ~/contribution/aiter   # or wherever this checkout lands on the AMD box
git fetch origin && git checkout feat/triton-batched-gemm-smallB-blockscale
git pull

# Confirm you're actually on gfx942/cu304 before spending time:
python3 -c "from aiter.jit.utils.chip_info import get_gfx; print(get_gfx())"
rocm-smi --showproductname 2>/dev/null | grep -i "MI300\|304"

# Run the tuner. Writes to a SCRATCH output first -- do not let it clobber
# the tracked CSV until you've reviewed the diff.
python3 csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py \
    -i aiter/configs/a8w8_blockscale_untuned_gemm.csv \
    -o /tmp/a8w8_blockscale_tuned_gemm.gfx942.csv

# Review: does it actually produce rows for the 6 target shapes, on gfx942/304?
grep -E "^gfx942,304,(32768|2048)," /tmp/a8w8_blockscale_tuned_gemm.gfx942.csv

# If it looks right: base_tuner.py (line ~422) already does proper merge
# logic internally (pd.concat of old rows not matched by the new run + the
# new/updated rows) -- confirmed by reading the source, not assumed. Safe to
# re-run pointing -o directly at the tracked file instead of a scratch path,
# but do the scratch-path run first anyway so `git diff` shows exactly what
# changed before it's committed.
python3 csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py \
    -i aiter/configs/a8w8_blockscale_untuned_gemm.csv \
    -o aiter/configs/a8w8_blockscale_tuned_gemm.csv
git diff --stat aiter/configs/a8w8_blockscale_tuned_gemm.csv
```

## Before opening a PR against ROCm/aiter for this

Per standing policy: check the issue thread again for any maintainer comment
that landed since 2026-07-11, and don't push without reviewing it. This is a
much lower-risk contribution than the small-B kernel (purely additive,
empirically-measured tuned-config data, not a new kernel design that might
not win anywhere) — but the same "read the room first" discipline applies.

## Priority note

This is the *bonus* task for tomorrow's AMD hardware time, after the small-B
kernel benchmark (`op_tests/bench_batched_gemm_smallB.py` +
`op_tests/test_batched_gemm_smallB.py`, both shape families: the original B=2
issue #3000 scope and the H>=8/M>=2048 regime zufayu endorsed on ATOM#960).
