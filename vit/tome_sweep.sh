#!/bin/bash
#SBATCH -J vit-tome
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
set -euo pipefail

echo "=== job $SLURM_JOB_ID on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

source /share/home/HAF9D47L/work/.venv-bikit/bin/activate
echo "python: $(which python)"

cd /share/home/HAF9D47L/work/project-aida/vit

python tome_sweep.py --r "${R_LIST:-0,4,8,12,16,24,32,40,48,56,64}" --out results

echo "=== done $(date) ==="
