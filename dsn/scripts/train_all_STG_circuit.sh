#!/bin/bash
for c_init in 0 5
do
  for rs in {1..2}
  do
    for sigma_init in 1.0 2.0 3.0
    do 
      sbatch train_STG_circuit.sh med 10 $c_init $sigma_init $rs
      sbatch train_STG_circuit.sh high 10 $c_init $sigma_init $rs
    done
  done
done
