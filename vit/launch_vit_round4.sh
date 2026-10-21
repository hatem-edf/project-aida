#!/bin/bash
# Round 4, run DETACHED:  setsid nohup bash launch_vit_round4.sh > launch_round4.log 2>&1 < /dev/null &
# Chain: MLP-only pruning (50/65/80%) -> 9 fine-tunes (100 ep, 3 seeds) -> ToMe on all pruned ViTs
#        -> final benchmark of every ViT and ResNet-50 variant (results/bench_all.{md,png,json}).
set -euo pipefail
cd /share/home/HAF9D47L/work/project-aida/vit
for f in runs/vit_sparsity/s50.pth runs/vit_sparsity/s65.pth runs/vit_sparsity/s80.pth \
         ../resnet50/prune_slim.py ../resnet50/paper_table.py; do
  [ -e "$f" ] || { echo "STOP: $f missing"; exit 1; }
done
for d in pruned/vit_mlp_r05.pth runs/vit_prune_mlp_r05_ep100_seed0; do
  [ -e "$d" ] && { echo "STOP: $d already exists"; exit 1; }
done
J1=$(sbatch --parsable prune_mlp_vit.sh)
FT=()
for S in 0 1 2; do
  for R in 05 065 08; do
    FT+=($(sbatch --parsable --dependency=afterok:$J1 \
           --export=ALL,CKPT=pruned/vit_mlp_r$R.pth,EXPECT=none,GUARD=20,EPOCHS=100,SEED=$S,TAG=prune_mlp_r${R}_ep100_seed$S train_vit.sh))
  done
done
DEP=$(IFS=:; echo "${FT[*]}")
J3=$(sbatch --parsable --dependency=afterok:$DEP tome_all.sh)
J4=$(sbatch --parsable --dependency=afterok:$J3 bench_all.sh)
{ echo; echo "## Round 4 launched $(date '+%F %T')"
  echo "MLP-only prune $J1 -> fine-tune ${FT[*]} -> ToMe on all pruned $J3 -> benchmark $J4"; } | tee -a HANDOFF.md
