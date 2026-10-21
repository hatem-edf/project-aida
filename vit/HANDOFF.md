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

## Round 3 — gated width pruning, launched 2026-10-20

### Step 1 — install
`install_round3.sh` md5 `18de4b477402bed5f7614a423a2a7203` (matches); installer ended
"install OK: 13 files written and verified". No code modified.

### Step 2 — login-node checks (all pass)
1. `eval_vit.py --golden-only`: **PASS** (worst 0.005 pts) — new any-width attention/MLP
   forwards are exact on the released weights.
2. `runs/vit_ctrl_ep100_seed0` max val EMR: **0.77597** (expected 0.776...).
3. `sparsity_vit.py --cpu-smoke`: 2 epochs, saved **s15 s30 s50 s65 s80** as required.
4. `prune_vit.py --ckpt s50.pth --ratio 0.5`: heads/block [1,1,1,1,1,1,1,5,6,6,6,6] (36/72),
   neurons/block floor 154, params 21,666,091 -> 11,040,043 (**49.0% smaller**),
   GFLOPs 44.8 -> 22.5, "saved". (All gates were 0 after the smoke's huge shrink, so the
   per-block floors dominate — as designed.)
5. `train_vit.py --cpu-smoke --ckpt pruned/vit_r05.pth`: end-to-end OK, ended
   "best -> /tmp/vit_ft_smoke/best.pth".
6. `import vit_table` OK.  /tmp dirs removed.

### Step 3 — commit
Round-3 code committed (19 vit files; a first commit again swept resnet50/ via a bare
`git add -A` and was corrected with `git reset HEAD~1 -- resnet50` + amend).

### Chain launched (detached)
sparsity (60 ep from seed0 best, expect val EMR 77.60) -> prune 15/30/50/65/80%
(each from its snapshot, else last.pth) -> 15 fine-tunes (5 ratios x 3 seeds, 100 ep,
GUARD=20) -> vit_table with measured GPU speed + ResNet-50 reference.
ETA ~12 h.

## Return checklist (fill in when back online)
1. `sacct -u $USER -S today -X --format=JobID,JobName%12,State,Elapsed,ExitCode`;
   for failures: exit code + last 20 log lines.
2. Sparsity log: per-epoch lines at ep 1, 5, 10, 20, 30, 40, 50, 60 (val EMR, heads off %,
   neurons off %) + epoch each snapshot was saved.
3. Prune log per ratio: snapshot used, heads/neurons per block, params, GFLOPs, val EMR gated -> pruned.
4. Per ratio+seed: best val EMR + epoch, test EMR.
5. `results/vit_table.md` verbatim.
Do not launch anything new until the senior reviews.

## Pruning launched 2026-10-20 16:01:00
sparsity 33073 -> prune 33074 -> fine-tune 33075 33076 33077 33078 33079 33080 33081 33082 33083 33084 33085 33086 33087 33088 33089 -> table 33090

## Prune fine-tunes relaunched 2026-10-20 18:21:08 (after cancellation of 33075-33090)
fine-tune 33094 33095 33096 33097 33098 33099 33100 33101 33102 33103 33104 33105 33106 33107 33108 -> table 33109

## Round 3 results (filled in 2026-10-20 evening)

All 16 resumed jobs (33094-33109) COMPLETED exit 0.
Sparsity: snapshots at ep 27 (s15), 30 (s30), 35 (s50), 39 (s65), 48 (s80).
Only MLP-neuron gates went to 0 (88.1% by ep 60); head gates never hit 0
(min mean 0.332) — all head removals came from ranking small nonzero gates.
Pruned-before-ft val EMR collapsed at high ratios (r05 1.46, r065/r08 0.81),
hence GUARD=20 was the right call.
Best val / test per ratio (3 seeds):
  r015: val 75.65/75.65/76.95  test 77.69/77.06/78.16
  r03:  val 75.65/75.81/75.65  test 75.63/74.21/76.74
  r05:  val 75.00/75.49/75.81  test 75.79/75.00/75.16
  r065: val 70.62/71.59/70.94  test 71.99/71.84/71.20
  r08:  val 66.72/68.02/67.21  test 70.41/69.62/68.04
Final table: results/vit_table.md (80% pruned ViT = ResNet FLOPs but -10 EMR pts;
ResNet still 1.7x faster in measured img/s).

## Round 4 — MLP-only pruning + ToMe on pruned + final bench, launched 2026-10-21

### Step 1 — install
install_round4.sh md5 954c33a5137c730160831f94fae4b3cb (matches); "install OK: 5 files
written and verified". No existing file changed.

### Step 2 — checks (all pass)
1. bench_all imports OK.
2. tome_tokens FLOPs reproduce the sweeps exactly: [44.8, 32.3, 21.2] at r=0/32/64.
3. MLP-only prune (s50, ratio 0.5, ratio-heads 0): heads 72/72, neurons 9216/18432
   (identical neuron selection to round 3), params 21.67M -> 14.58M (**32.7% smaller**,
   GFLOPs -24.8%; the brief's "about 25%" is the GFLOPs figure), "saved".
4. ResNet references present: 5 pruned b160 ckpts + ft_control_ep160_seed0/best.pth,
   15 ft_b160 test jsons.
5. tmp cleaned.

### Chain launched (detached)
MLP-only prune 50/65/80% -> 9 fine-tunes (3 seeds x 100 ep) -> ToMe r=0,16,32,48,64
on all 24 pruned ViTs -> bench_all final table + plot. ETA ~6 h.

## Return checklist (fill in when back online)
1. sacct job states; for failures: exit code + last 20 log lines.
2. MLP-only prune per ratio: val EMR gated -> pruned, params, GFLOPs — side by side
   with round 3 heads+MLP (r05: 69.64 -> 1.46, 11.04M, 22.5 GFLOPs;
   r065: 71.10 -> 0.81, 7.83M, 15.7; r08: 71.43 -> 0.81, 4.62M, 8.9).
3. Per MLP-only ratio: test EMR mean +/- std over 3 seeds.
4. results/bench_all.md verbatim + Pareto-marked models.
5. Send Hatem results/bench_all.png.
After this round the ViT work stops; write-up follows.

## Round 4 launched 2026-10-21 10:57:42
MLP-only prune 33111 -> fine-tune 33112 33113 33114 33115 33116 33117 33118 33119 33120 -> ToMe on all pruned 33121 -> benchmark 33122

## Round 4 PAUSED at user request (GPUs needed) — state snapshot 2026-10-21

Completed: 33111 MLP-only prune (pruned/vit_mlp_r05/r065/r08.pth + stats); fine-tunes
33112-33119 = 8 of 9 runs, all COMPLETED with results/vit_prune_mlp_*_test.json.
Cancelled by user: 33120 = **vit_prune_mlp_r08_ep100_seed2** (6 min in; its run dir holds
~7 epochs of partial best/last/history), 33121 ToMe-all, 33122 bench.

### Resume procedure (when told)
1. rm -rf runs/vit_prune_mlp_r08_ep100_seed2   (partial dir of the cancelled run)
2. sbatch the one missing fine-tune:
   CKPT=pruned/vit_mlp_r08.pth EXPECT=none GUARD=20 EPOCHS=100 SEED=2 TAG=prune_mlp_r08_ep100_seed2 train_vit.sh
3. afterok: tome_all.sh (ToMe r=0,16,32,48,64 on all 24 pruned ViTs)
4. afterok: bench_all.sh (results/bench_all.{md,png,json})
5. Report per the round-4 return checklist above.

## Round 4 resumed 2026-10-21 18:21:56 (after user pause; 33120/33121/33122 cancelled)
fine-tune r08 s2 33125 -> ToMe on all pruned 33126 -> benchmark 33127
