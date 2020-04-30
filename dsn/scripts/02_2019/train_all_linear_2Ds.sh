#!/bin/bash

for reps in 2
do
  for c0 in -3
  do
    for rs in {1..10}
    do
      sbatch train_linear_2D.sh $reps $c0 1.0 $rs
      sbatch train_linear_2D.sh $reps $c0 3.0 $rs
    done
  done
done
