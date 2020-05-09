#!/bin/bash
for alpha in E P S V
do
  for stages in 2
  do
    for units in 50
    do
      for logc0 in 0
      do
        for rs in {1..4}
        do
          sbatch run_V1_dr.sh $alpha $stages $units $logc0 $rs
        done
      done
    done
  done
done
