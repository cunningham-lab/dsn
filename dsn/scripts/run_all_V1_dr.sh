#!/bin/bash
for alpha in E
do
  for beta in P S
  do
    for inc_std in 0.25
    do
      for stages in 2 3
      do
        for units in 50
        do
          for rs in {1..5}
          do
            sbatch run_V1_dr.sh $alpha $beta $inc_std $stages $units $rs
          done
        done
      done
    done
  done
done
