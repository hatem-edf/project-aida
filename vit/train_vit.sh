#!/bin/bash
#SBATCH -J vit-train
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
set -euo pipefail

echo "=== job $SLURM_JOB_ID on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
source /share/home/HAF9D47L/work/.venv-bikit/bin/activate
echo "python: $(which python)"
cd /share/home/HAF9D47L/work/project-aida/vit

TAG="${TAG:?set TAG}"
EPOCHS="${EPOCHS:?set EPOCHS}"
SEED="${SEED:-0}"
CKPT="${CKPT:-$HOME/.cache/bikit-models/codebrim-classif-balanced_ViT_s8_1.pth}"
EXPECT="${EXPECT:-76.30}"          # val EMR of the released model; EXPECT=none for other starts
G1=""
[ "$EXPECT" != none ] && G1="--expect-val-emr $EXPECT"

python train_vit.py --ckpt "$CKPT" --epochs "$EPOCHS" --seed "$SEED" $G1 --out "runs/vit_${TAG}"
python eval_vit.py --ckpt "runs/vit_${TAG}/best.pth" --tag "vit_${TAG}" --no-golden --split test --out results

echo "=== done $(date) ==="
