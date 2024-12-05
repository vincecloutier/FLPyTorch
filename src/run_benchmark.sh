#!/bin/bash

# Default values
PROCESSES=8
RUNS=3
DATASETS="cifar"

# Parse arguments
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --processes)
    PROCESSES="$2"
    shift # past argument
    shift # past value
    ;;
    --runs)
    RUNS="$2"
    shift
    shift
    ;;
    --datasets)
    DATASETS="$2"
    shift
    shift
    ;;
    *)
    echo "Unknown option $1"
    exit 1
    ;;
esac
done

echo "Using $PROCESSES processes."
echo "Number of runs: $RUNS"
echo "Datasets: $DATASETS"

# Loop for the number of runs
for i in $(seq 1 $RUNS)
do
    # Split datasets by comma and loop through each
    for dataset in $(echo $DATASETS | tr ',' ' ')
    do
        echo "Run $i for $dataset"
        # Run commands for settings 0 to 3
        for setting in {0..3}
        do
            python benchmark.py --dataset $dataset --setting $setting --processes $PROCESSES
        done
    done
done

echo "All runs completed."

# chmod +x run_benchmark.sh
# ./run_benchmark.sh --processes 8 --runs 3 --datasets cifar,fmnist