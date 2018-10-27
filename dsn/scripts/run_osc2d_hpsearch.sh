#!/bin/bash
  
counter=1

for c in -5 0 5
do
  for lr in -2 -3 -4
  do
    nohup python3 osc2D_hpsearch_helper.py 1 0 $c $lr 2>&1 > $counter.log &
    ((counter++))
    nohup python3 osc2D_hpsearch_helper.py 0 2 $c $lr 2>&1 > $counter.log &
    ((counter++))
    nohup python3 osc2D_hpsearch_helper.py 0 4 $c $lr 2>&1 > $counter.log &
    ((counter++))
    nohup python3 osc2D_hpsearch_helper.py 0 8 $c $lr 2>&1 > $counter.log &
    ((counter++))
    nohup python3 osc2D_hpsearch_helper.py 0 10 $c $lr 2>&1 > $counter.log &
    ((counter++))
  done
done

echo All done!
