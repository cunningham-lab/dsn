#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=V1_dr
#SBATCH -c 1
#SBATCH --time=3:59:00
#SBATCH --mem-per-cpu=10gb

source activate epi
python3 V1_dr.py --alpha $1 --inc_val 0. --inc_std 0.25 --num_stages $2 --num_units $3 --logc0 $4 --random_seed $5
