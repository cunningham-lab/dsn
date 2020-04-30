#!/bin/bash
for c_init in 0 5 10
do
  for rs in {1..5}
  do
    sbatch train_V1_diff.sh E 0 $c_init $rs
    sbatch train_V1_diff.sh E 1 $c_init $rs
  done
done
