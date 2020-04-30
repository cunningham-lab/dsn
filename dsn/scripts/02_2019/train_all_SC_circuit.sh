#!/bin/bash
for rs in {1..5}
do
  for sigma_init in 1.0 3.0
  do
    sbatch train_SC_circuit.sh full 0 $rs 3 $sigma_init
    sbatch train_SC_circuit.sh full 0 $rs 5 $sigma_init
  done
done
