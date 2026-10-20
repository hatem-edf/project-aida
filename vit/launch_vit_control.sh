#!/bin/bash
# One-shot launch of the ViT fine-tuning control. Submitting frees the GPUs (llm-stack goes down),
# so run it DETACHED, it must finish even if the shell that started it dies:
#   setsid nohup bash launch_vit_control.sh > launch.log 2>&1 < /dev/null &
# Chain: 2-epoch GPU smoke (guards, speed, memory) -> 3 seeds x 100 epochs -> ToMe on each.
set -euo pipefail
cd /share/home/HAF9D47L/work/project-aida/vit
for d in runs/vit_smoke_gpu runs/vit_ctrl_ep100_seed0 runs/vit_ctrl_ep100_seed1 runs/vit_ctrl_ep100_seed2; do
  [ -e "$d" ] && { echo "STOP: $d already exists"; exit 1; }
done
mkdir -p logs results runs
J1=$(sbatch --parsable --export=ALL,EPOCHS=2,SEED=0,TAG=smoke_gpu train_vit.sh)
C=()
for S in 0 1 2; do
  C+=($(sbatch --parsable --dependency=afterok:$J1 --export=ALL,EPOCHS=100,SEED=$S,TAG=ctrl_ep100_seed$S train_vit.sh))
done
J5=$(sbatch --parsable --dependency=afterok:${C[0]}:${C[1]}:${C[2]} --export=ALL,STEM=ctrl_ep100 tome_ctrl.sh)
{ echo; echo "## Launched $(date '+%F %T')"
  echo "smoke $J1 -> control ${C[*]} (seeds 0 1 2) -> ToMe $J5"; } | tee -a HANDOFF.md
