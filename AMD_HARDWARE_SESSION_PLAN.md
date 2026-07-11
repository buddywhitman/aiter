# AMD hardware session plan — everything to run, in order

Supersedes `systems/prs/project4_smallb_fp8_aiter/AMD_BENCHMARK_INSTRUCTIONS.md`,
which is stale (wrong repo path, wrong file names, predates the NaN fix, the
retargeted shapes, and the #3852 bonus task). That file now just points here.

Two independent tasks, run in this order:
1. **Small-B kernel benchmark** — the real question: does the kernel win anywhere.
2. **ROCm/aiter#3852 tuned-config gap** — lower-risk bonus, real customer ask.

---

## 0. Instance provisioning

You already have the $200 AMD Developer Cloud credits, so skip straight to
launching an instance:

1. https://developer.amd.com/tools/cloud/ → **Instances** → **Launch Instance**
2. GPU: **MI300X (gfx942)** if available — it's the one both tasks target
   directly (issue #3000's shapes, #3852's `cu_num=304`). If only **MI355X
   (gfx950)** is offered, that's fine too — the kernel fix now handles gfx950's
   `float8_e4m3fn` correctly, but note in your results which arch you actually
   ran on, since gfx942 vs gfx950 numbers aren't interchangeable for #3852
   specifically (that issue is gfx942-only).
3. Image: **ROCm 6.x (or later) + PyTorch pre-installed**, if offered — saves
   a slow from-scratch PyTorch/ROCm install. 1 GPU is enough for both tasks.
4. Add your SSH public key under **Settings → SSH Keys** before launching.
5. Note the public IP once it's up.

Verify access and confirm the arch before doing anything else — don't burn
GPU time on a mis-provisioned instance:

```bash
ssh -i /path/to/key.pem ubuntu@<AMD_IP>
rocminfo | grep -i gfx        # expect gfx942 or gfx950
rocm-smi --showproductname    # expect "MI300X" or "MI355X"
```

## 1. Clone the branch (avoid the embedded-PAT pattern)

The `origin` remote on your local machine's checkout has a PAT embedded
directly in the URL (`https://github_pat_...@github.com/...`) — don't copy
that URL onto the new AMD box. Use `gh` or SSH instead:

```bash
# On the AMD instance:
gh auth login   # if gh isn't already authenticated there
gh repo clone buddywhitman/aiter -- -b feat/triton-batched-gemm-smallB-blockscale
cd aiter
git log --oneline -3
# Should show d0eac9d13 (aiter#3852 prep) at the top -- if not, `git pull`.
```

If `gh` isn't available on the image: `git clone -b feat/triton-batched-gemm-smallB-blockscale https://github.com/buddywhitman/aiter.git`
(public repo, no auth needed for a plain clone).

Set git identity once, so any commits made here don't repeat the
wrong-email-resolves-to-a-different-GitHub-account issue from earlier sessions:

```bash
git config user.name "Pulkit Kumar"
git config user.email "buddywhitman@users.noreply.github.com"
```

## 2. Environment setup

Both tasks need PyTorch with a ROCm build. Confirm what the image gives you
before installing anything:

```bash
python3 -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
```

If `torch.version.hip` is `None`, install the ROCm PyTorch build:
```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2   # match the installed ROCm version
```

**The two tasks need different aiter install modes** — this is the single
biggest thing the old doc got wrong by omission:

- **Task 1 (Triton kernel benchmark)** only needs the Triton ops. Use
  `AITER_TRITON_ONLY=1` to skip the full CK/HIP C++ extension build entirely —
  much faster, and this kernel has zero CK dependency.
- **Task 2 (#3852 tuning)** needs the *compiled* `ck_gemm_a8w8_blockscale`
  CK extension — the tuner profiles real compiled kernel latency, not Triton.
  Do **not** set `AITER_TRITON_ONLY=1` for this task; let the full build run.

```bash
# Editable install, no build yet (fast) -- confirms the package structure works:
pip install -e . --no-build-isolation
```

If a full build is needed for Task 2 later and takes a long time, that's
expected (CK has a lot of kernel instances to compile) — budget for it, don't
assume it hung.

## 3. Task 1 — Small-B kernel benchmark

### 3a. Correctness first (fast, catches anything environment-specific)

```bash
cd ~/aiter   # wherever you cloned it
AITER_TRITON_ONLY=1 AITER_AMD_FP8=1 PYTHONPATH=. python op_tests/test_batched_gemm_smallB.py
```

Expect: `All 13 tests PASSED.` with `FP8 dtype: torch.float8_e4m3fnuz (AMD)` (or
`e4m3fn` if the tuner resolves to gfx950) printed at the top — **not**
`float8_e5m2`. If it prints e5m2, `AITER_AMD_FP8=1` didn't take effect; check
the env var actually made it into the process (some shells drop inline env
vars across `sudo`/`su` — export it instead if that happens:
`export AITER_AMD_FP8=1`).

If this fails with NaN-related errors, stop and report back before spending
more GPU time — that would mean the fix from `85188d4ba` didn't fully resolve
it, which is a real, unexpected finding worth an immediate check-in rather
than continuing to burn credits.

### 3b. Benchmark both shape families

```bash
AITER_TRITON_ONLY=1 AITER_AMD_FP8=1 PYTHONPATH=. python op_tests/bench_batched_gemm_smallB.py 2>&1 | tee ~/amd_bench_results.txt
```

This covers, in one run:
- The original issue #3000 shapes (B=2, M=1..1024) — including the explicit
  M=512/TP=8 shape from ROCm/ATOM#960 (zufayu's production shape, where the
  *different* `fp8_einsum` kernel lost 1.9x to BF16).
- The H≥8/M≥2048 "compute-bound" regime zufayu said was the more promising
  target — this is the number that actually answers whether this kernel
  family is worth pursuing at all.

Save `~/amd_bench_results.txt` — you'll paste real numbers from it into PR #2
and the ATOM#960 thread, replacing the "NVIDIA directional, not authoritative"
caveats currently there.

### 3c. What to do with the results

- Update `buddywhitman/aiter` PR #2's description with real numbers (I can do
  this via the REST API once you have the numbers — same pattern as the
  NVIDIA-directional update).
- Per your earlier decision: post the real numbers into the live
  **ROCm/ATOM#960** thread (zufayu/ganyi1996ppo), since that's where this
  exact shape is being actively discussed — not a cold PR against
  `ROCm/aiter` issue #3000.
- **Do not** open a PR against the real `ROCm/aiter` repo yet regardless of
  outcome. If it wins: that's when the "review the thread, confirm alignment"
  step actually matters — post in ATOM#960 first, see if zufayu or anyone
  responds, *then* decide on a real PR. If it loses: say so plainly in PR #2
  and the ATOM#960 thread; that's still a useful, honest data point for
  whoever's tracking this problem next, and matches the "don't overclaim"
  discipline from this session's PR-description fixes.

## 4. Task 2 — ROCm/aiter#3852 (bonus, after Task 1)

Full runbook already written: `ROCM_AITER_3852_RUNBOOK.md` in this same repo
root. Short version:

```bash
cd ~/aiter
# Confirm you're actually on gfx942/cu304 -- #3852 is specifically about that arch:
python3 -c "from aiter.jit.utils.chip_info import get_gfx; print(get_gfx())"
rocm-smi --showproductname | grep -i "304\|MI300"

python3 csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py \
    -i aiter/configs/a8w8_blockscale_untuned_gemm.csv \
    -o /tmp/a8w8_blockscale_tuned_gemm.gfx942.csv

grep -E "^gfx942,304,(32768|2048)," /tmp/a8w8_blockscale_tuned_gemm.gfx942.csv
```

If that grep shows rows for all 6 shapes (3 prefill at M=32768, 3 decode at
M=2048), the tuner worked. Copy the result back and diff against the tracked
CSV before committing:

```bash
scp -i /path/to/key.pem ubuntu@<AMD_IP>:/tmp/a8w8_blockscale_tuned_gemm.gfx942.csv ./tuned_gfx942_result.csv
# back on your local machine:
diff aiter/configs/a8w8_blockscale_tuned_gemm.csv tuned_gfx942_result.csv | head -50
```

If you're running this on the AMD box directly and just want it to land in
the tracked file there, `base_tuner.py`'s merge logic (confirmed by reading
the source, `aiter/utility/base_tuner.py:422`) correctly preserves existing
rows — safe to point `-o` straight at `aiter/configs/a8w8_blockscale_tuned_gemm.csv`
instead of the scratch path, then `git diff --stat` to see what changed.

This is lower-risk than Task 1: it's empirically-measured tuned-config data,
not a kernel design whose win is unproven. Per the runbook, check the issue
thread again for any maintainer comment before opening a PR against the real
`ROCm/aiter` — but a clean, small, purely-additive CSV diff closing a real
customer-reported gap is about as safe a PR as this kind of contribution gets.

## 5. Cost/time budget

| Step | Est. time |
|---|---|
| Instance boot | 2–5 min |
| PyTorch/ROCm check + editable install (Triton-only) | 2–5 min |
| Task 1 correctness (3a) | ~2 min |
| Task 1 benchmark, both shape families (3b) | ~10–15 min |
| Task 2 full aiter build (needed for CK) | 15–40 min (real CK build, budget for it) |
| Task 2 tuner run, 6 shapes | ~5–15 min (profiles real kernel latency) |
| **Total GPU time** | **~45–80 min** |

Shut the instance down as soon as both tasks' results are saved locally —
nothing left after that needs the GPU running.
