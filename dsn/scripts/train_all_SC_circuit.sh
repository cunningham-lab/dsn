#!/bin/bash
for p in 0.8
do
  for c_init in 5 15
  do
    for rs in {1..5}
    do
      sbatch train_SC_circuit.sh $p full $c_init $rs
    done
  done
done
