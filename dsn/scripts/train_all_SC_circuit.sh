#!/bin/bash
for nlayers in 5 10
do
  for sigma_init in 1.0
  do
    for c_init in -10 -5 0 5
    do
      for rs in {1..5}
      do
        sbatch train_SC_circuit.sh $nlayers $c_init $sigma_init $rs
      done
    done
  done
done
