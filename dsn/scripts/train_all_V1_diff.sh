#!/bin/bash
for c_init in -5 0 5
do
  for rs in {1..5}
  do
    for sigma_init in 1.0 5.0
    do
      sbatch train_V1_diff.sh E 0 $c_init $sigma_init $rs
      sbatch train_V1_diff.sh E 1 $c_init $sigma_init $rs
    done
  done
done
