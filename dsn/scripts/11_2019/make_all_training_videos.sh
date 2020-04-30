#!/bin/bash

for c_init in 0 1
do
  for K in 1 2 4 16
  do
    for sigma_init in 1 2 3
    do
      for rs in {1..5}
      do
        sbatch make_training_video.sh $1 10 $c_init $K $sigma_init $rs
      done
    done
  done
done
