# HANDOFF — ViT round 2 (fine-tuning control)

Agent session of 2026-10-20. Code copied from the senior's brief, checked on the login node,
committed, then launched detached.

## Step 0 — quick checks (all pass)

```
$ sacct -j 33057 --format=JobID,State,ExitCode,Elapsed
JobID             State    ExitCode  Elapsed
33057             COMPLETED 0:0      00:02:05
33057.batch       COMPLETED 0:0      00:02:05
33057.extern      COMPLETED 0:0      00:02:05
```
(The kill observed in the previous round was not a failure: state COMPLETED, exit 0.)

- `results/tome_sweep.json`: **11 rows** (expected 11).
- Crack decisions flipped vs r=0 (test, threshold 0.5):
  `r4 0  r8 1  r12 3  r16 2  r24 3  r32 6  r40 8  r48 8  r56 11  r64 10` (of 632).

## Step 1 — files (copied byte-for-byte from the brief, except one fix)

- NEW `train_vit.py`, `train_vit.sh`, `tome_ctrl.sh`, `launch_vit_control.sh`
- EDIT `eval_vit.py`: `--no-golden` flag; golden test skipped when set.
- EDIT `tome_sweep.py`: `--ckpt` argument, passed to `load_vit()`.
- **Deviation (1):** the brief's G0 line ended `...permutations(range(len(perm)))})` —
  unmatched `)`, a `SyntaxError` at import. I removed the extra `)` (line 151).
  Everything else is untouched.

## Step 2 — login-node checks (all pass)

1. `bash -n`: `train_vit.sh OK`, `tome_ctrl.sh OK`, `launch_vit_control.sh OK`.
   `python -m py_compile` OK on all three .py files.
2. `python train_vit.py --cpu-smoke --out /tmp/vit_smoke`:
   train 100 / val 100 (devel_mode placeholders), 21,666,055 trainable params,
   1 epoch in 55 s, ended with `best -> /tmp/vit_smoke/best.pth`. (1m39 total)
3. `python eval_vit.py --golden-only`: **PASS** (worst 0.005 pts, tolerance 0.5) —
   default path of the edited file unaffected.
4. `python eval_vit.py --ckpt /tmp/vit_smoke/best.pth --tag smoke --no-golden --split val
   --workers 0 --out /tmp/vit_smoke_res`: checkpoint loads in the bikit-models layout,
   full val eval runs, `smoke_val.json` + `_raw.npz` written. (Numbers meaningless by design.)
5. `python tome_sweep.py --help | grep -- --ckpt`: flag present.
6. `/tmp/vit_smoke*` removed.

Compatibility probe: `BikitDataset(..., devel_mode=True)` exists in the cluster's bikit.

## Step 3 — commit

`df3a1df` "ViT round 2: fine-tuning control code" — the 6 vit files only.
(A first commit accidentally swept the uncommitted resnet50/ work via a repo-root `git add -A`;
undone with `git reset HEAD~1 -- resnet50` + amend. resnet50/ is back in its exact prior state.)

## Launched

`setsid nohup bash launch_vit_control.sh > launch.log 2>&1 < /dev/null &`

## Return checklist (fill in when back online)

1. `sacct -u $USER -S today -X --format=JobID,JobName%12,State,Elapsed,ExitCode` — every job;
   for failures: which guard (exit 2/3/4/5/6) + last 20 lines of the log.
2. From the smoke log: G0 line, "before fine-tuning" (val EMR must be 76.30), secs/epoch, peak memory.
3. Per seed: best val EMR + epoch; val EMR at epochs 1, 5, 10, 25, 50, 100;
   test EMR / F1 / per-class recall from `results/vit_ctrl_ep100_seed*_test.json`.
4. Mean ± std of test EMR over the 3 seeds vs released ViT 79.27 @ 44.8 GFLOPs
   and fine-tuned ResNet-50 79.32 @ 8.2 GFLOPs.
5. Per-seed ToMe sweep: val/test EMR at r = 0, 16, 32, 64 with speedup.
   (`results/tome_ctrl_ep100_seed{0,1,2}/tome_sweep.json`)

Do not launch anything new until the senior reviews these results.

## Launched 2026-10-20 11:41:08
smoke 33066 -> control 33067 33068 33069 (seeds 0 1 2) -> ToMe 33070

## Return checklist — filled in 2026-10-20 (all jobs COMPLETED, exit 0)

- Jobs: 33066 smoke 2m32s; 33067/33068/33069 seeds 1h25m each; 33070 ToMe 5m09s. No guards tripped.
- Smoke: G0 our order 0.1284 = best of 720 (no-reorder 0.3979); before ft val EMR 76.30 (G1 ok);
  ~47 s/epoch; peak 13.2 GB.
- Best val EMR: seed0 77.60 @ ep84, seed1 77.44 @ ep53, seed2 77.27 @ ep18.
  Val EMR at ep 1/5/10/25/50/100:
  seed0: 75.65 / 75.00 / 74.19 / 74.35 / 76.79 / 77.11
  seed1: 76.14 / 73.38 / 75.16 / 74.19 / 76.62 / 75.97
  seed2: 73.86 / 73.38 / 75.00 / 73.38 / 75.32 / 75.49
- Test (best ckpts): seed0 79.59 (F1 89.65), seed1 79.11 (F1 89.48), seed2 79.75 (F1 90.06).
  **mean +/- std = 79.48 +/- 0.33** vs released ViT 79.27 @ 44.8 GFLOPs, ft ResNet-50 79.32 @ 8.2 GFLOPs.
  Test recall: Cr 89.3-93.3, Ef 81.9-83.2, Ru 85.3-86.0, ND 95.3-96.7, Sp 88.0-88.7, BE 92.7-93.3.
- ToMe on each best ckpt (val/test EMR, speedup):
  r=0:  77.60/79.59, 77.44/79.11, 77.27/79.75  (44.8 GFLOPs, 284-288 img/s)
  r=16: 76.46/79.11, 77.92/79.27, 76.62/80.06  (38.5 GFLOPs, ~1.02x)
  r=32: 76.46/78.96, 76.79/79.11, 75.97/79.59  (32.3 GFLOPs, ~1.26x)
  r=64: 75.16/78.32, 75.65/77.69, 74.51/78.01  (21.2 GFLOPs, ~1.97x)

## Round 3 blocker (2026-10-20)

`install_round3.sh` not present in ~/work/project-aida/vit — nothing installed,
nothing checked, nothing launched. Waiting for the file (scp or nvim paste).
Expected md5: 18de4b477402bed5f7614a423a2a7203.
