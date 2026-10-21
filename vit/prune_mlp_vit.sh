#!/bin/bash
#SBATCH -J vit-prune-mlp
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
# MLP-only pruning (all 72 heads kept) from the existing sparsity snapshots.
set -euo pipefail
echo "=== job $SLURM_JOB_ID on $(hostname) at $(date) ==="
source /share/home/HAF9D47L/work/.venv-bikit/bin/activate
cd /share/home/HAF9D47L/work/project-aida/vit
for R in 0.5 0.65 0.8; do
  N=$(python -c "print(f'{round(100 * $R):02d}')")
  SNAP="runs/vit_sparsity/s$N.pth"
  [ -f "$SNAP" ] || { echo "STOP: $SNAP missing"; exit 1; }
  echo; echo "########## MLP-only ratio $R from $SNAP ##########"
  python prune_vit.py --ckpt "$SNAP" --ratio "$R" --ratio-heads 0 --eval-val --out "pruned/vit_mlp_r${R/./}.pth"
done
echo "=== done $(date) ==="
