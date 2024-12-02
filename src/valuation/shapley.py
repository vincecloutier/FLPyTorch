import os
import numpy as np
from update import test_inference
from utils import average_weights, initialize_model, get_device
from collections import defaultdict
from math import factorial as fact
import itertools
import multiprocessing
import torch


_global_weights = None
_client_weights = None
_test_dataset = None


def init_process(global_weights, client_weights, test_dataset):
    global _global_weights, _client_weights, _test_dataset
    _global_weights = global_weights
    _client_weights = client_weights
    _test_dataset = test_dataset


def compute_shapley(args, global_weights, client_weights, test_dataset):
    """Estimate Shapley values for participants in a round using permutation sampling."""
    device = get_device()

    # initialize model and compute base accuracy
    model = initialize_model(args)
    model.load_state_dict(global_weights)
    model.to(device)

    with torch.no_grad():
        base_acc = test_inference(model, test_dataset)[0]

    client_keys = list(client_weights.keys())
    epsilon, delta, r = 0.25, 0.25, 1  # allow 25% error at 75% confidence
    t = int((2 * r**2 / epsilon**2) * np.log(2 * len(client_keys) / delta))

    shapley_updates = defaultdict(float)
    if t < fact(len(client_keys)):
        args_list = [(None, client_keys, base_acc, device, args) for _ in range(t)]
    else:
        args_list = [(perm, client_keys, base_acc, device, args) for perm in itertools.permutations(client_keys)]
    pool = multiprocessing.Pool(processes=args.shapley_processes, initializer=init_process, initargs=(global_weights, client_weights, test_dataset))
    results = pool.map(compute_shapley_for_permutation, args_list)
    pool.close()
    pool.join()

    for result in results:
        for k, v in result.items():
            shapley_updates[k] += v

    del results
    torch.cuda.empty_cache()

    # average the values over all permutations
    shapley_updates = {k: v / t for k, v in shapley_updates.items()}
    return shapley_updates


def compute_shapley_for_permutation(args):
    (perm, client_keys, base_acc, device, args_model) = args

    model = initialize_model(args_model)
    model.load_state_dict(_global_weights)
    model.to(device)

    shapley_updates_local = defaultdict(float)

    if perm is None:
        perm = np.random.permutation(client_keys)

    prev_acc = base_acc
    current_weights = []
    for i in perm:
        current_weights.append(_client_weights[i])
        avg_weights = average_weights(current_weights)
        model.load_state_dict(avg_weights)
        with torch.no_grad():
            curr_acc = test_inference(model, _test_dataset)[0]
        shapley_updates_local[i] += curr_acc - prev_acc
        prev_acc = curr_acc

    del model
    torch.cuda.empty_cache()
    
    return shapley_updates_local