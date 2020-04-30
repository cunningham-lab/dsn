#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=V1_dr
#SBATCH -c 1
#SBATCH --time=1:59:00
#SBATCH --mem-per-cpu=10gb

source activate epi
python3 V1_dr.py --alpha E --inc_val 0. --inc_std 0.01 --num_stages $1 --num_units $2 --logc0 0 --random_seed $3
