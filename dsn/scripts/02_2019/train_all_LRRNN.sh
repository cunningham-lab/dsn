#!/bin/bash
for repeats in 1 2
do
  for sigma_init in 1.0 2.0
  do
    for rs in {1..5}
    do
      sbatch train_LRRNN.sh $repeats $sigma_init 0.1 0 $rs
    done
  done
done
