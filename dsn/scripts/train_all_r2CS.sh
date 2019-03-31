#!/bin/bash
for nlayers in 10
do
  for sigma_init in 1.0
  do
    for c_init in -15 -10 -5 0
    do
      for rs in {1..5}
      do
        sbatch train_rank2_CDD_static.sh $nlayers $c_init $sigma_init $rs
      done
    done
  done
done
