#!/bin/bash

# get processes argument
if [ -z "$1" ]; then
    PROCESSES=8  # default is eight
else
    PROCESSES=$1
fi

echo "Using $PROCESSES processes."

# loop to run each command three times
for i in {1..3}
do

    # CIFAR commands
    echo "Run $i for CIFAR"
    python benchmark.py --dataset cifar --setting 0 --processes $PROCESSES
    python benchmark.py --dataset cifar --setting 1 --processes $PROCESSES
    python benchmark.py --dataset cifar --setting 2 --processes $PROCESSES
    python benchmark.py --dataset cifar --setting 3 --processes $PROCESSES

    # FMNIST commands
    # echo "Run $i for FMNIST"
    # python benchmark.py --dataset fmnist --setting 0 --processes $PROCESSES
    # python benchmark.py --dataset fmnist --setting 1 --processes $PROCESSES
    # python benchmark.py --dataset fmnist --setting 2 --processes $PROCESSES
    # python benchmark.py --dataset fmnist --setting 3 --processes $PROCESSES

    # IMAGENET commands
    # echo "Run $i for IMAGENET"
    # python benchmark.py --dataset imagenet --setting 0 --processes $PROCESSES
    # python benchmark.py --dataset imagenet --setting 1 --processes $PROCESSES
    # python benchmark.py --dataset imagenet --setting 2 --processes $PROCESSES
    # python benchmark.py --dataset imagenet --setting 3 --processes $PROCESSES

done

echo "All runs completed."

# chmod +x run_benchmark.sh
# ./run_benchmark.sh [processes]