#!/bin/bash
for p in 0.9
do
  for c in 2 
  do
    for rs in {1..5}
    do
      for upl in 10 15
      do
        sbatch train_SC_WTA.sh full $p NI $c $upl 1.0 $rs
      done
    done
  done
done
