#!/bin/bash

for D in 3 4
do
  for sigma_init in 1.0 10.0 100.0
  do
    sbatch init_nfs.sh $D 20 $sigma_init
  done
done
