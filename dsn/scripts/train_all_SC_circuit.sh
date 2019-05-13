#!/bin/bash
for c_init in 0 3 4 5
do
  for rs in {1..4}
  do
    sbatch train_SC_circuit.sh full $c_init $rs
    sbatch train_SC_circuit.sh reduced $c_init $rs
  done
done
