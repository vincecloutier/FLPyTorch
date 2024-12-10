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
    --local_ep)
    LOCAL_EP="$2"
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
echo "Local Epochs: $LOCAL_EP"

for i in $(seq 1 $RUNS)
do
    for dataset in $(echo $DATASETS | tr ',' ' ')
    do
        echo "Run $i for $dataset"
        for setting in {0..3}
        do
            if [ $dataset == "fmnist" ]; then
                ACCURACY_STOPPING=0
            else
                ACCURACY_STOPPING=1
            fi
            python benchmark.py --dataset $dataset --setting $setting --processes $PROCESSES --acc_stopping $ACCURACY_STOPPING --local_ep $LOCAL_EP
        done
    done
done

echo "All runs completed."

# chmod +x run_benchmark.sh
# ./run_benchmark.sh --processes 8 --datasets cifar,fmnist --local_ep 3