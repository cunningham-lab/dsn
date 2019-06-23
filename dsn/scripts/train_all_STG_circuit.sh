#!/bin/bash
for K in 10 20
do
  for rs in {1..2}
  do
    for sigma_init in 2.0 3.0
    do 
      for sigma0 in 0.1
      do
        sbatch train_STG_circuit.sh 10 2 $K $sigma_init $rs $sigma0
      done
    done
  done
done
