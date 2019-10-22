#!/bin/bash
for repeats in 1
do
  for sigma_init in 1.0
  do
    for rs in 2
    do
      sbatch train_LRRNN.sh $repeats $sigma_init 0.01 0 $rs
    done
  done
done
