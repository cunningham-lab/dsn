#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=V1_dr
#SBATCH -c 1
#SBATCH --gres=gpu
#SBATCH --time=1:59:00
#SBATCH --mem-per-cpu=2gb

module load cuda90/toolkit
module load cuda90/blas
module load cudnn/7.0.5

source activate epi_gpu
python3 V1_dr.py --alpha E --inc_val 0. --inc_std 0.01 --num_stages $1 --num_units $2 --logc0 0 --random_seed $3
