#!/bin/bash
for c_init in -2 0 2 4
do
  for rs in {1..2}
  do
    for sigma_init in 3.0
    do 
      sbatch train_STG_circuit.sh med 10 $c_init 4 $sigma_init $rs
    done
  done
done
