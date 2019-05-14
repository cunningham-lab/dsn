#!/bin/bash
for rs in {1..10}
do
  sbatch train_SC_circuit.sh full 1 $rs
  sbatch train_SC_circuit.sh reduced 1 $rs
done
