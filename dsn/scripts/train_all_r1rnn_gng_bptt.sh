#!/bin/bash
for rs in {0..1499}
do
  sbatch train_r1rnn_gng_bptt.sh $rs
done
