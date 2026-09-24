#!/bin/bash
#SBATCH -J slim-sparse
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
set -euo pipefail
echo "=== $SLURM_JOB_ID on $(hostname) $(date) ==="
source /share/home/HAF9D47L/work/.venv-bikit/bin/activate
cd /share/home/HAF9D47L/work/project-aida/resnet50

LAM="${LAM:-5e-5}"
EPOCHS="${EPOCHS:-40}"
CKPT="${CKPT:-$HOME/.cache/bikit-models/codebrim-classif-balanced_ResNet50_hta.pth}"
TAG="${TAG:-lam${LAM}}"

echo "base checkpoint: $CKPT"
python sparsity_train.py --ckpt "$CKPT" --lam "$LAM" --epochs "$EPOCHS" \
       --out "runs/sparsity_${TAG}"
echo "=== done $(date) ==="
