#!/bin/bash

for c_init_ord in -4 -2 0
do
  for nlayers in 2
  do
    for sigma_init in 3.0
    do
      for rs in {1..4}
      do
        sbatch train_linear_2D.sh $c_init_ord $nlayers $sigma_init $rs
      done
    done
  done
done
