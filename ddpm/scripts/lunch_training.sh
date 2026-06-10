#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

#SBATCH --job-name=sentinel_fb_gen
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err
#SBATCH --time=23:55:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

python "${REPO_ROOT}/models/Imagen_text_pytorch.py" "$@"
# accelerate launch "${REPO_ROOT}/models/Imagen_text_pytorch.py" "$@"

