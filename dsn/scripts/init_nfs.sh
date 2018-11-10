#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=init_nf
#SBATCH --gres=gpu
#SBATCH -c 1
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=1gb

source activate dsn
python3 init_nfs.py $1 $2 $3 $4
