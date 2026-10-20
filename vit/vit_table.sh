#!/bin/bash
#SBATCH -J vit-table
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
set -euo pipefail
source /share/home/HAF9D47L/work/.venv-bikit/bin/activate
cd /share/home/HAF9D47L/work/project-aida/vit
python vit_table.py ${PREVIEW:+--preview}
