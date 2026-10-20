#!/bin/bash
#SBATCH -J vit-prune
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
set -euo pipefail
echo "=== job $SLURM_JOB_ID on $(hostname) at $(date) ==="
source /share/home/HAF9D47L/work/.venv-bikit/bin/activate
cd /share/home/HAF9D47L/work/project-aida/vit

SP="runs/vit_${SP_TAG:-sparsity}"
mkdir -p pruned
for R in ${RATIOS:-0.15 0.3 0.5 0.65 0.8}; do
  N=$(python -c "print(f'{round(100 * $R):02d}')")          # 0.15 -> 15, 0.3 -> 30
  SNAP="$SP/s$N.pth"
  [ -f "$SNAP" ] || { echo "!! $SNAP not reached, pruning from last.pth"; SNAP="$SP/last.pth"; }
  echo; echo "########## ratio $R from $SNAP ##########"
  python prune_vit.py --ckpt "$SNAP" --ratio "$R" --eval-val --out "pruned/vit_r${R/./}.pth"
done
echo "=== done $(date) ==="
