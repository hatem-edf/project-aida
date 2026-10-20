#!/bin/bash
#SBATCH -J vit-tome
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
set -euo pipefail

echo "=== job $SLURM_JOB_ID on $(hostname) at $(date) ==="
source /share/home/HAF9D47L/work/.venv-bikit/bin/activate
cd /share/home/HAF9D47L/work/project-aida/vit

STEM="${STEM:-ctrl_ep100}"
for S in 0 1 2; do
  echo; echo "########## ToMe on vit_${STEM}_seed$S ##########"
  python tome_sweep.py --ckpt "runs/vit_${STEM}_seed$S/best.pth" --out "results/tome_${STEM}_seed$S"
done
echo "=== done $(date) ==="
