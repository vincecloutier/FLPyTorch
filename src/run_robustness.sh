#!/bin/bash

# get processes and shapley_processes arguments
if [ -z "$1" ]; then
    PROCESSES=8  # default is eight
else
    PROCESSES=$1
fi

if [ -z "$2" ]; then
    SHAPLEY_PROCESSES=6  # default is six
else
    SHAPLEY_PROCESSES=$2
fi

echo "Using $PROCESSES processes."

# python robustness.py --dataset cifar --setting 0 --processes $PROCESSES --num_users 10 --local_ep 10 --epochs 15 --shapley_processes $SHAPLEY_PROCESSES
python robustness.py --dataset cifar --setting 1 --processes $PROCESSES --num_users 10 --local_ep 10 --epochs 15 --shapley_processes $SHAPLEY_PROCESSES
# python robustness.py --dataset cifar --setting 2 --processes $PROCESSES --num_users 10 --local_ep 10 --epochs 15 --shapley_processes $SHAPLEY_PROCESSES
# python robustness.py --dataset cifar --setting 3 --processes $PROCESSES --num_users 10 --local_ep 10 --epochs 15 --shapley_processes $SHAPLEY_PROCESSES

echo "All runs completed."

# chmod +x run_robustness.sh
# ./run_robustness.sh [processes] [shapley_processes]
