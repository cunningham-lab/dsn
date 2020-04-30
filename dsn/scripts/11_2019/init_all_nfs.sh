#!/bin/bash

for D in 8 
do
  for sigma_init in 1.0
  do
    for rs in {7..100}
    do
      sbatch init_nfs.sh $D 10 $sigma_init $rs
    done
  done
done
