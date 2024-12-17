#!/bin/bash

# get processes argument
if [ -z "$1" ]; then
    PROCESSES=8  # default is eight
else
    PROCESSES="$1"
fi

if [ -z "$2" ]; then
    LR=1e-2  # default is 1e-2
else
    LR="$2"
fi

if [ -z "$3" ]; then
    DATASET="resnet"
else
    DATASET="$3"
fi

if [ -z "$4" ]; then
    LOCAL_EP=10
else
    LOCAL_EP="$4"
fi

# install bc if not installed
if ! command -v bc &> /dev/null; then
    echo "'bc' is not installed. Attempting to install it..."
    apt-get update && apt-get install -y bc
else
    echo "'bc' is already installed."
fi

# loop to run each command nine times
for i in {1..4}
do
    # calculate badclient_prop for this iteration
    BAD_CLIENT_PROP=$(echo "scale=1; 0.2 * $i" | bc)

    echo "Run $i with bad_client_prop=$BAD_CLIENT_PROP"
    # python retraining.py --dataset $DATASET --setting 1 --processes $PROCESSES --badclient_prop $BAD_CLIENT_PROP --num_users 10 --local_ep $LOCAL_EP --retrain 1 --num_categories_per_client 4 --lr $LR
    # python retraining.py --dataset $DATASET --setting 2 --processes $PROCESSES --badclient_prop $BAD_CLIENT_PROP --num_users 10 --local_ep $LOCAL_EP --retrain 1 --badsample_prop 0.6 --lr $LR
    python retraining.py --dataset $DATASET --setting 3 --processes $PROCESSES --badclient_prop $BAD_CLIENT_PROP --num_users 10 --local_ep $LOCAL_EP --retrain 1 --badsample_prop 1 --lr $LR

    # python retraining.py --dataset $DATASET --setting 1 --processes $PROCESSES --badclient_prop $BAD_CLIENT_PROP --num_users 10 --local_ep $LOCAL_EP --retrain 1 --num_categories_per_client 6 --lr $LR
    # python retraining.py --dataset $DATASET --setting 2 --processes $PROCESSES --badclient_prop $BAD_CLIENT_PROP --num_users 10 --local_ep $LOCAL_EP --retrain 1 --badsample_prop 0.4 --lr $LR
    python retraining.py --dataset $DATASET --setting 3 --processes $PROCESSES --badclient_prop $BAD_CLIENT_PROP --num_users 10 --local_ep $LOCAL_EP --retrain 1 --badsample_prop 0.875 --lr $LR

    # python retraining.py --dataset $DATASET --setting 1 --processes $PROCESSES --badclient_prop $BAD_CLIENT_PROP --num_users 10 --local_ep $LOCAL_EP --retrain 1 --num_categories_per_client 8 --lr $LR
    # python retraining.py --dataset $DATASET --setting 2 --processes $PROCESSES --badclient_prop $BAD_CLIENT_PROP --num_users 10 --local_ep $LOCAL_EP --retrain 1 --badsample_prop 0.2 --lr $LR
    python retraining.py --dataset $DATASET --setting 3 --processes $PROCESSES --badclient_prop $BAD_CLIENT_PROP --num_users 10 --local_ep $LOCAL_EP --retrain 1 --badsample_prop 0.75 --lr $LR

done

echo "All runs completed."

# chmod +x run_retraining.sh
# ./run_retraining.sh [processes] [lr] [dataset] [local_ep]