#!/bin/bash

# loop to run each command 3 times
for i in {1..3}
do
    echo "Run $i for FMNIST"

    # FMNIST commands
    python benchmark.py --dataset fmnist --setting 0 --processes 30 --local_ep 10
    python benchmark.py --dataset fmnist --setting 1 --processes 30 --local_ep 3
    python benchmark.py --dataset fmnist --setting 2 --processes 30 --local_ep 3
    python benchmark.py --dataset fmnist --setting 3 --processes 30 --local_ep 3

    echo "Run $i for IMAGENET"

    # IMAGENET commands
    python benchmark.py --dataset imagenet --setting 0 --processes 30 --local_ep 10
    python benchmark.py --dataset imagenet --setting 1 --processes 30 --local_ep 3
    python benchmark.py --dataset imagenet --setting 2 --processes 30 --local_ep 3
    python benchmark.py --dataset imagenet --setting 3 --processes 30 --local_ep 3

done

echo "All runs completed."

# chmod +x run_benchmark.sh
# ./run_benchmark.sh