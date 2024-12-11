#!/bin/bash

# get processes and shapley_processes and dataset arguments
if [ -z "$1" ]; then
    PROCESSES=8 
else
    PROCESSES=$1
fi

if [ -z "$2" ]; then
    SHAPLEY_PROCESSES=6  
else
    SHAPLEY_PROCESSES=$2
fi

if [ -z "$3" ]; then
    DATASET=cifar
else
    DATASET=$3
fi

if [ -z "$4" ]; then
    LOCAL_EP=10
else
    LOCAL_EP=$4
fi

echo "Using $PROCESSES processes."

python robustness.py --dataset $DATASET --setting 0 --processes $PROCESSES --num_users 10 --local_ep $LOCAL_EP --epochs 15 --shapley_processes $SHAPLEY_PROCESSES
python robustness.py --dataset $DATASET --setting 1 --processes $PROCESSES --num_users 10 --local_ep $LOCAL_EP --epochs 15 --shapley_processes $SHAPLEY_PROCESSES
python robustness.py --dataset $DATASET --setting 2 --processes $PROCESSES --num_users 10 --local_ep $LOCAL_EP --epochs 15 --shapley_processes $SHAPLEY_PROCESSES
python robustness.py --dataset $DATASET --setting 3 --processes $PROCESSES --num_users 10 --local_ep $LOCAL_EP --epochs 15 --shapley_processes $SHAPLEY_PROCESSES

echo "All runs completed."

# chmod +x run_robustness.sh
# ./run_robustness.sh [processes] [shapley_processes] [dataset] [local_ep]