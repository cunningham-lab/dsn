#!/bin/bash
for c_init_ord in 2
do
  for sigma_init in 0.5
  do
    for rs in {1..5}
    do
      sbatch train_STG_circuit.sh 4 2 $c_init_ord $sigma_init $rs
    done
  done
done
