#!/bin/bash
for c_init in 5
do
  for rs in {1..4}
  do
    for alpha in E P S V 
    do
      sbatch train_V1_diff.sh $alpha 0.1 $c_init $rs
      sbatch train_V1_diff.sh $alpha 0.5 $c_init $rs
    done
  done
done
