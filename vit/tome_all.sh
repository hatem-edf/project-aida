#!/bin/bash
#SBATCH -J vit-tome-all
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
# ToMe sweep on every fine-tuned pruned ViT (all seeds): pruning + merging combined.
set -euo pipefail
echo "=== job $SLURM_JOB_ID on $(hostname) at $(date) ==="
source /share/home/HAF9D47L/work/.venv-bikit/bin/activate
cd /share/home/HAF9D47L/work/project-aida/vit
for D in runs/vit_prune_r*_ep100_seed* runs/vit_prune_mlp_r*_ep100_seed*; do
  NAME=$(basename "$D")
  OUT="results/tome_$NAME"
  [ -f "$OUT/tome_sweep.json" ] && { echo "skip $NAME (done)"; continue; }
  echo; echo "########## ToMe on $NAME ##########"
  python tome_sweep.py --ckpt "$D/best.pth" --r 0,16,32,48,64 --out "$OUT"
done
echo "=== done $(date) ==="
