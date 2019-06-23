#!/bin/bash
for c_init in 2
do
  for rs in {1..2}
  do
    for sigma_init in 3.0 5.0
    do 
      for sigma0 in 0.1 0.5 1.0
      do
        sbatch train_STG_circuit.sh 10 $c_init 4 $sigma_init $rs $sigma0
      done
    done
  done
done
