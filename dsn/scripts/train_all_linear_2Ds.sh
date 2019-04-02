#!/bin/bash

for num_masks in 2 4
do
  for nlayers in 2 4
  do
    for upl in 5 10 20
    do
      for rs in {1..5}
      do
        sbatch train_linear_2D.sh $num_masks $nlayers $upl $rs
      done
    done
  done
done
