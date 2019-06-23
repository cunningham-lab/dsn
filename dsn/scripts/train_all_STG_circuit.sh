#!/bin/bash
for c_init in 1 2
do
  for rs in {1..2}
  do
    for sigma_init in 1.0 3.0
    do 
      sbatch train_STG_circuit.sh 10 $c_init 4 $sigma_init $rs
    done
  done
done
