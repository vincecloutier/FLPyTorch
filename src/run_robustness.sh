#!/bin/bash

# get processes argument
if [ -z "$1" ]; then
    PROCESSES=8  # default is eight
else
    PROCESSES=$1
fi

echo "Using $PROCESSES processes."

# loop to run each command nine times
for i in {1..9}
do
    # calculate noise_std for this iteration
    NOISE_STD=$(echo "scale=1; 0.1 * $i" | bc)

    # CIFAR commands
    echo "Run $i for CIFAR with noise_std=$NOISE_STD"
    python robustness.py --dataset cifar --setting 0 --processes $PROCESSES --noise_std $NOISE_STD --num_users 50 --local_ep 3
    python robustness.py --dataset cifar --setting 1 --processes $PROCESSES --noise_std $NOISE_STD --num_users 50 --local_ep 3
    python robustness.py --dataset cifar --setting 2 --processes $PROCESSES --noise_std $NOISE_STD --num_users 50 --local_ep 3
    python robustness.py --dataset cifar --setting 3 --processes $PROCESSES --noise_std $NOISE_STD --num_users 50 --local_ep 3

    # FMNIST commands
    echo "Run $i for FMNIST with noise_std=$NOISE_STD"
    python robustness.py --dataset fmnist --setting 0 --processes $PROCESSES --noise_std $NOISE_STD --num_users 50 --local_ep 3
    python robustness.py --dataset fmnist --setting 1 --processes $PROCESSES --noise_std $NOISE_STD --num_users 50 --local_ep 3
    python robustness.py --dataset fmnist --setting 2 --processes $PROCESSES --noise_std $NOISE_STD --num_users 50 --local_ep 3
    python robustness.py --dataset fmnist --setting 3 --processes $PROCESSES --noise_std $NOISE_STD --num_users 50 --local_ep 3

    # IMAGENET commands
    echo "Run $i for IMAGENET with noise_std=$NOISE_STD"
    python robustness.py --dataset imagenet --setting 0 --processes $PROCESSES --noise_std $NOISE_STD --num_users 50 --local_ep 3
    python robustness.py --dataset imagenet --setting 1 --processes $PROCESSES --noise_std $NOISE_STD --num_users 50 --local_ep 3
    python robustness.py --dataset imagenet --setting 2 --processes $PROCESSES --noise_std $NOISE_STD --num_users 50 --local_ep 3
    python robustness.py --dataset imagenet --setting 3 --processes $PROCESSES --noise_std $NOISE_STD --num_users 50 --local_ep 3

done

echo "All runs completed."

# chmod +x run_robustness.sh
# ./run_robustness.sh [processes]
