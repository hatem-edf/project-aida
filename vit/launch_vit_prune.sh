#!/bin/bash
# One-shot launch of ViT width pruning, run DETACHED (submitting frees the GPUs):
#   setsid nohup bash launch_vit_prune.sh > launch_prune.log 2>&1 < /dev/null &
# Chain: sparsity training (gates + L1) -> prune at 15/30/50/65/80% -> 100-epoch fine-tuning,
#        3 seeds each (same budget as the control) -> final table with measured throughput.
set -euo pipefail
cd /share/home/HAF9D47L/work/project-aida/vit
START=runs/vit_ctrl_ep100_seed0/best.pth          # control seed with the best val EMR
EXPECT=77.60                                      # its val EMR
RATIOS="0.15 0.3 0.5 0.65 0.8"
[ -f "$START" ] || { echo "STOP: $START missing"; exit 1; }
for d in runs/vit_sparsity pruned/vit_r015.pth runs/vit_prune_r015_ep100_seed0; do
  [ -e "$d" ] && { echo "STOP: $d already exists"; exit 1; }
done
grep -q 'GUARD=' train_vit.sh || { echo "STOP: train_vit.sh lacks the GUARD option"; exit 1; }

J1=$(sbatch --parsable --export=ALL,CKPT=$START,EXPECT=$EXPECT,TAG=sparsity sparsity_vit.sh)
J2=$(sbatch --parsable --dependency=afterok:$J1 --export=ALL,SP_TAG=sparsity prune_all_vit.sh)
FT=()
for S in 0 1 2; do
  for R in $RATIOS; do
    N=r${R/./}
    FT+=($(sbatch --parsable --dependency=afterok:$J2 \
           --export=ALL,CKPT=pruned/vit_$N.pth,EXPECT=none,GUARD=20,EPOCHS=100,SEED=$S,TAG=prune_${N}_ep100_seed$S train_vit.sh))
  done
done
DEP=$(IFS=:; echo "${FT[*]}")
J4=$(sbatch --parsable --dependency=afterok:$DEP vit_table.sh)
{ echo; echo "## Pruning launched $(date '+%F %T')"
  echo "sparsity $J1 -> prune $J2 -> fine-tune ${FT[*]} -> table $J4"; } | tee -a HANDOFF.md
