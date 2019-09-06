#!/bin/bash
for p in 0.5 0.6 0.7 0.9
do
  for c in 2
  do
    for rs in {6..10}
    do
      sbatch train_SC_WTA.sh full $p NI $c 1.0 $rs
    done
  done
done
