#!/bin/bash
  
counter=1

for D in 10
do
  for c in -5
  do
    for lr in -2
    do
      for T in 10
      do 
        nohup python3 rank1rnn_hpsearch_helper.py $D $D $c $lr $T 2>&1 > $counter.log &
        ((counter++))
      done
    done
  done
done

#for D in 10 25
#do
#  for c in -5 5
#  do
#    for lr in -2 -3 -4
#    do
#      for T in 10 15
#      do 
#        nohup python3 rank1rnn_hpsearch_helper.py $D $D $c $lr $T 2>&1 > $counter.log &
#        ((counter++))
#      done
#    done
#  done
#done

echo All done!
