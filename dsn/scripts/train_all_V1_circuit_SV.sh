#!/bin/bash
for nlayers in 10
do
  for sigma_init in 1.0
  do
    for c_init in -5 0 5
    do
      for rs in {1..5}
      do
        sbatch train_V1_circuit_SV.sh $nlayers $c_init $sigma_init $rs
      done
    done
  done
done
