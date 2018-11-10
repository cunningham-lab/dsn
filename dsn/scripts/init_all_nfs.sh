#!/bin/bash

for D in 3 4
do
  for sigma_init in 1.0 10.0 100.0
  do
    for rs in 1 2 3 4 5 
    do
      sbatch init_nfs.sh $D 10 $sigma_init $rs
      sbatch init_nfs.sh $D 20 $sigma_init $rs
    done
  done
done
