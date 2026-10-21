#!/bin/bash
# Resume round 4 after the user pause (2026-10-21): one missing fine-tune
# (MLP-only r08, seed 2) -> ToMe on all pruned ViTs -> final benchmark.
# Same parameters as launch_vit_round4.sh. Run DETACHED:
#   setsid nohup bash resume_round4.sh > resume_round4.log 2>&1 < /dev/null &
set -euo pipefail
cd /share/home/HAF9D47L/work/project-aida/vit
[ -f pruned/vit_mlp_r08.pth ] || { echo "STOP: pruned/vit_mlp_r08.pth missing"; exit 1; }
for R in 05 065 08; do
  for S in 0 1 2; do
    [ "$R$S" = "082" ] && continue
    [ -f "results/vit_prune_mlp_r${R}_ep100_seed${S}_test.json" ] || { echo "STOP: no test json for r${R} seed${S}"; exit 1; }
  done
done
[ -f results/vit_prune_mlp_r08_ep100_seed2_test.json ] && { echo "STOP: r08 seed2 already done"; exit 1; }
[ -e runs/vit_prune_mlp_r08_ep100_seed2 ] && { echo "STOP: runs/vit_prune_mlp_r08_ep100_seed2 exists (remove it first)"; exit 1; }
J1=$(sbatch --parsable --export=ALL,CKPT=pruned/vit_mlp_r08.pth,EXPECT=none,GUARD=20,EPOCHS=100,SEED=2,TAG=prune_mlp_r08_ep100_seed2 train_vit.sh)
J3=$(sbatch --parsable --dependency=afterok:$J1 tome_all.sh)
J4=$(sbatch --parsable --dependency=afterok:$J3 bench_all.sh)
{ echo; echo "## Round 4 resumed $(date '+%F %T') (after user pause; 33120/33121/33122 cancelled)"
  echo "fine-tune r08 s2 $J1 -> ToMe on all pruned $J3 -> benchmark $J4"; } | tee -a HANDOFF.md
