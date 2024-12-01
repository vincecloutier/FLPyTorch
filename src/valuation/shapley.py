import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import numpy as np
from update import test_inference
from utils import average_weights, initialize_model, get_device
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def compute_shapley_for_permutation(args):
    (client_keys, client_weights, global_weights, base_acc, test_dataset, device, args_model) = args
    permutation = np.random.permutation(client_keys)
    prev_acc = base_acc

    # initize model inside the function
    model = initialize_model(args_model)
    model.load_state_dict(global_weights)
    model.to(device)
    model.train()

    current_weights = []
    shapley_updates_local = defaultdict(float)
    for i in permutation:
        current_weights.append(client_weights[i])
        avg_weights = average_weights(current_weights)
        model.load_state_dict(avg_weights)
        curr_acc = test_inference(model, test_dataset)[0]
        shapley_updates_local[i] += curr_acc - prev_acc
        prev_acc = curr_acc
    return shapley_updates_local

def compute_shapley(args, global_weights, client_weights, test_dataset):
    """Estimate Shapley values for participants in a round using permutation sampling."""
    device = get_device()

    # Initialize model and compute base accuracy
    model = initialize_model(args)
    model.load_state_dict(global_weights)
    model.to(device)
    model.train()
    base_acc = test_inference(model, test_dataset)[0]

    client_keys = list(client_weights.keys())
    m = len(client_keys)
    epsilon, delta, r = 0.25, 0.25, 1  # Allow 25% error at 75% confidence
    t = int((2 * r**2 / epsilon**2) * np.log(2 * m / delta))

    shapley_updates = defaultdict(float)

    # Prepare arguments for parallel execution
    args_list = [
        (client_keys, client_weights, global_weights, base_acc, test_dataset, device, args)
        for _ in range(t)
    ]

    with ProcessPoolExecutor(max_workers=args.shapley_processes) as executor:
        futures = [executor.submit(compute_shapley_for_permutation, arg) for arg in args_list]

        # Use tqdm to display progress
        for future in tqdm(as_completed(futures), total=t, desc="Calculating Shapley Values"):
            shapley_update_local = future.result()
            for k, v in shapley_update_local.items():
                shapley_updates[k] += v

    # Average the Shapley values over all permutations
    shapley_updates = {k: v / t for k, v in shapley_updates.items()}
    return shapley_updates
