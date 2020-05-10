#!/bin/bash
for alpha in E
do
  for inc_std in 0.25
  do
    for stages in 2
    do
      for units in 50
      do
        for logc0 in 0
        do
          for rs in {1..4}
          do
            sbatch run_V1_dr.sh $alpha $inc_std $stages $units $logc0 $rs
          done
        done
      done
    done
  done
done
