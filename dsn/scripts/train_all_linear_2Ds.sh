#!/bin/bash

for nlayers in 10
do
  for c_order in -10 -5 0
  do
    for sigma_init in 1 10
    do
      for rs in {1..5}
      do
        sbatch train_linear_2D.sh $nlayers $c_order $sigma_init $rs
      done
    done
  done
done
