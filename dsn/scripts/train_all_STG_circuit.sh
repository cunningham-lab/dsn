#!/bin/bash
for num_masks in 2 4
do
  for rs in {1..4}
  do
    sbatch train_STG_circuit.sh $num_masks 2 2 $rs
  done
done
