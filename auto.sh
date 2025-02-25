#!/bin/sh
# create a loop to run the auto_eval_model.py script for different values of n, waiting for the previous one to finish before starting the next one
# max=9
# for i in `seq 4 $max`
# do
#     python auto_eval_model.py tqdn3 $i
#     sleep 1
# done

max_top_k=10
for i in `seq 2 $max_top_k`
do
    python auto_eval_model.py tqdn3 5 $i
    sleep 1
done