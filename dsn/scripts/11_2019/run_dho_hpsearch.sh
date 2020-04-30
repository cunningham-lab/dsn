#!/bin/bash
  
counter=1

for c in -10 0 10
do
  for lr in -3 -4
  do
    for T in 10 30 50 
    do 
      nohup python3 dho_hpsearch_helper.py 5 $c $lr $T 2>&1 > $counter.log &
      ((counter++))
      nohup python3 dho_hpsearch_helper.py 10 $c $lr $T 2>&1 > $counter.log &
      ((counter++))
    done
  done
done

echo All done!
