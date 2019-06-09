#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=movie
#SBATCH -c 1
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=500Mb

source activate dsn
python3 make_training_video.py $1 $2 $3 $4 $5 $6
