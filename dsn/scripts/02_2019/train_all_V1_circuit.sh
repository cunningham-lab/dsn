#!/bin/bash
for repeats in 1
do
  for nlayers in 2 
  do
    for upl in 10 15
    do
      for rs in {1..5}
      do
        sbatch train_V1_circuit.sh 3.0 $repeats $nlayers $upl 5 $rs
        sbatch train_V1_circuit.sh 5.0 $repeats $nlayers $upl 5 $rs
      done
    done
  done
done
