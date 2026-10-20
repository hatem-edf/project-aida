#!/bin/bash
#SBATCH -J vit-sparse
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
set -euo pipefail
echo "=== job $SLURM_JOB_ID on $(hostname) at $(date) ==="
source /share/home/HAF9D47L/work/.venv-bikit/bin/activate
cd /share/home/HAF9D47L/work/project-aida/vit

CKPT="${CKPT:?set CKPT (the fine-tuned control best.pth)}"
EXPECT="${EXPECT:?set EXPECT (val EMR of that checkpoint, e.g. 77.60)}"
python sparsity_vit.py --ckpt "$CKPT" --expect-val-emr "$EXPECT" --epochs "${EPOCHS:-60}" \
       --lam-head "${LAM:-2e-4}" --lam-mlp "${LAM:-2e-4}" --lr-gate "${LR_GATE:-2e-3}" --out "runs/vit_${TAG:-sparsity}"
echo "=== done $(date) ==="
