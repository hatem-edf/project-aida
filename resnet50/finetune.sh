#!/bin/bash
#SBATCH -J slim-ft
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
set -euo pipefail

echo "=== $SLURM_JOB_ID on $(hostname) $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source /share/home/HAF9D47L/work/.venv-bikit/bin/activate
echo "python: $(which python)"
cd /share/home/HAF9D47L/work/project-aida/resnet50

CKPT="${CKPT:?set CKPT=pruned/xxx.pth}"
SEED="${SEED:-0}"
TAG="$(basename "${CKPT%.pth}")_seed${SEED}"

python finetune_pruned.py \
    --ckpt "$CKPT" \
    --seed "$SEED" \
    --epochs "${EPOCHS:-40}" \
    --lr "${LR:-1e-4}" \
    --out "runs/ft_${TAG}"

# final test-set number for the recovered model
python eval_codebrim.py \
    --ckpt "runs/ft_${TAG}/best.pth" \
    --tag "ft_${TAG}" \
    --split test \
    --out results

echo "=== done $(date) ==="
