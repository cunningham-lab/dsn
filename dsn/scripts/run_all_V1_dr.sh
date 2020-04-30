#!/bin/bash
for stages in 2 3
do
  for units in 15 25
  do
    for rs in {1..5}
    do
      sbatch run_V1_dr.sh $stages $units $rs
    done
  done
done
