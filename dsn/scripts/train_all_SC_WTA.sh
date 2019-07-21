#!/bin/bash
for p in 0.5 0.6 0.7 0.8 0.9 1.0
do
  for c in 2 4
  do
    for rs in 1 2 3 4 5
    do
      sbatch train_SC_WTA.sh full $p NI $c 1.0 $rs
    done
  done
done
