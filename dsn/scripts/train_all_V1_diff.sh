#!/bin/bash
for c_init in 0
do
  for rs in {1..5}
  do
    for sigma_init in 1.0
    do
      sbatch train_V1_diff.sh pos V $c_init $sigma_init $rs
      sbatch train_V1_diff.sh neg V $c_init $sigma_init $rs
    done
  done
done
