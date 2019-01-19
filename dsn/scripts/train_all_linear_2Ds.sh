#!/bin/bash

for nlayers in 10 20
do
  for c_order in -10 0
  do
    for sigma_init in 1 10 100
    do
      sbatch test_linear_2D.sh $nlayers $c_order $sigma_init 0
    done
  done
done
