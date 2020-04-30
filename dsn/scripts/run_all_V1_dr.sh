#!/bin/bash
for stages in 2
do
  for units in 15
  do
    for rs in 1
    do
      sbatch run_V1_dr.sh $stages $units $rs
    done
  done
done
