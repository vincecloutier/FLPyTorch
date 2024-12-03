#!/bin/bash

# get processes argument
if [ -z "$1" ]; then
    PROCESSES=8  # default is eight
else
    PROCESSES=$1
fi

echo "Using $PROCESSES processes."

python robustness.py --dataset resnet --setting 0 --processes $PROCESSES --num_users 5 --local_ep 10
python robustness.py --dataset resnet --setting 1 --processes $PROCESSES --num_users 5 --local_ep 10
python robustness.py --dataset resnet --setting 2 --processes $PROCESSES --num_users 5 --local_ep 10
python robustness.py --dataset resnet --setting 3 --processes $PROCESSES --num_users 5 --local_ep 10

echo "All runs completed."

# chmod +x run_robustness.sh
# ./run_robustness.sh [processes]
