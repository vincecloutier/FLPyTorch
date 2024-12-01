import numpy as np
from update import test_inference
from utils import average_weights, initialize_model, get_device
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def compute_shapley_for_permutations(args):
    (client_keys, client_weights, global_weights, base_acc, test_dataset, device, args_model, num_permutations) = args
    shapley_updates_local = defaultdict(float)

    model = initialize_model(args_model)
    model.load_state_dict(global_weights)
    model.to(device) 
    model.train()

    for _ in tqdm(range(num_permutations), desc="Calculating Shapley Values"):
        permutation = np.random.permutation(client_keys)
        prev_acc = base_acc

        current_weights = []
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
  
    num_processes = args.shapley_processes
    args_list = []
    for i in range(num_processes):
        num_permutations = t // num_processes + (1 if i < (t % num_processes) else 0)
        process_args = (client_keys, client_weights, global_weights, base_acc, test_dataset, device, args, num_permutations)
        args_list.append(process_args)

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        shapley_update_local = executor.map(compute_shapley_for_permutations, args_list)
        for updates in shapley_update_local:
            for k, v in updates.items():
                shapley_updates[k] += v

    # Average the Shapley values over all permutations
    shapley_updates = {k: v / t for k, v in shapley_updates.items()}
    return shapley_updates
