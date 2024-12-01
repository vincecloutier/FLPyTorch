import torch
import numpy as np
import multiprocessing
from tqdm import tqdm
from collections import defaultdict
from update import LocalUpdate, test_inference
from utils import average_weights, initialize_model
import random
import copy
from functools import partial


def compute_monte_carlo_shapley(args, global_weights, train_dataset, user_groups, device, test_dataset):
    """Compute Monte Carlo Shapley values for clients."""
    num_clients = args.num_users
    shapley_values = defaultdict(float)
    num_samples = 2*args.num_users

    clients = list(range(num_clients))

    # parallelize the processing of random permutations
    with multiprocessing.Pool(processes=args.processes) as pool:
        results = list(tqdm(pool.imap(
            partial(process_permutation, args=args, global_weights=global_weights, train_dataset=train_dataset, 
                    user_groups=user_groups, device=device, test_dataset=test_dataset, clients=clients),
            range(num_samples)), total=num_samples, desc="Shapley Permutations"))
    # aggregate the shapley values
    for permutation_shapley in results:
        for client_idx, contribution in permutation_shapley.items():
            shapley_values[client_idx] += contribution
    
    # average the contributions
    for client_idx in shapley_values:
        shapley_values[client_idx] /= num_samples
    
    return shapley_values


def process_permutation(sample_idx, args, global_weights, train_dataset, user_groups, device, test_dataset, clients):
    random_permutation = copy.deepcopy(clients)
    random.shuffle(random_permutation)
    
    current_weights = copy.deepcopy(global_weights)
    current_model = initialize_model(args)
    current_model.load_state_dict(current_weights)
    current_model.to(device)
    current_model.train()
    
    # initial performance
    initial_perf = test_inference(current_model, test_dataset)[0]  # assuming higher is better
    
    permutation_shapley = defaultdict(float)
    
    for client_idx in random_permutation:
        # train the model with the current client
        local_update = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[client_idx])
        w, _ = local_update.update_weights(model=current_model, global_round=sample_idx)
        current_weights = average_weights([current_weights, w])
        current_model.load_state_dict(current_weights)
        
        # performance after adding the client
        new_perf = test_inference(current_model, test_dataset)[0]
        
        # marginal contribution
        marginal_contribution = new_perf - initial_perf
        permutation_shapley[client_idx] += marginal_contribution
        
        # update initial performance for next client in permutation
        initial_perf = new_perf
    
    del current_model
    torch.cuda.empty_cache()
    
    return permutation_shapley