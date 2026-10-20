#!/bin/bash
# Resume the round-3 chain after the user cancelled jobs 33075-33090 (2026-10-20).
# Sparsity (33073) and prune (33074) had COMPLETED, so only the 15 fine-tunes and the
# final table are resubmitted, with the same parameters as launch_vit_prune.sh.
# Run DETACHED (submitting frees the GPUs):
#   setsid nohup bash relaunch_prune_ft.sh > relaunch_prune_ft.log 2>&1 < /dev/null &
set -euo pipefail
cd /share/home/HAF9D47L/work/project-aida/vit
RATIOS="0.15 0.3 0.5 0.65 0.8"
for R in $RATIOS; do
  N=r${R/./}
  [ -f "pruned/vit_$N.pth" ] || { echo "STOP: pruned/vit_$N.pth missing"; exit 1; }
done
for S in 0 1 2; do
  for R in $RATIOS; do
    d="runs/vit_prune_r${R/./}_ep100_seed$S"
    [ -e "$d" ] && { echo "STOP: $d already exists"; exit 1; }
  done
done
FT=()
for S in 0 1 2; do
  for R in $RATIOS; do
    N=r${R/./}
    FT+=($(sbatch --parsable --export=ALL,CKPT=pruned/vit_$N.pth,EXPECT=none,GUARD=20,EPOCHS=100,SEED=$S,TAG=prune_${N}_ep100_seed$S train_vit.sh))
  done
done
DEP=$(IFS=:; echo "${FT[*]}")
J4=$(sbatch --parsable --dependency=afterok:$DEP vit_table.sh)
{ echo; echo "## Prune fine-tunes relaunched $(date '+%F %T') (after cancellation of 33075-33090)"
  echo "fine-tune ${FT[*]} -> table $J4"; } | tee -a HANDOFF.md
