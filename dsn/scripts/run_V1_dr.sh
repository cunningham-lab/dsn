#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=V1_dr
#SBATCH -c 1
#SBATCH --time=3:59:00
#SBATCH --mem-per-cpu=10gb

source activate epi
python3 V1_dr.py --alpha $1 --beta $2 --inc_val 0. --inc_std $3 --num_stages $4 --num_units $5 --random_seed $6
