import itertools
import math
from collections import defaultdict
import copy
import time
import torch
import numpy as np
from tqdm import tqdm
from options import args_parser
from update import LocalUpdate, test_inference, test_gradient
from models import CNNMnist, CNNFashion_Mnist, CNNCifar, ResNet9, MobileNetV2
from utils import get_dataset, average_weights, exp_details, setup_logger, get_device

from scipy.stats import pearsonr, spearmanr
import torch.multiprocessing as mp

def initialize_model(args, device):
    model_dict = {
        'mnist': CNNMnist,
        'fmnist': CNNFashion_Mnist,
        'cifar': CNNCifar,
        'resnet': ResNet9,
        'mobilenet': MobileNetV2
    }
    model_class = model_dict.get(args.dataset)
    if not model_class:
        exit('Error: unrecognized dataset')
    return model_class(args=args).to(device)

def train_global_model(args, model, train_dataset, test_dataset, user_groups, device, clients=None, isBanzhaf=False):
    if clients is None or len(clients) == 0:
        return model, defaultdict(float)
    
    global_weights = model.state_dict()
    best_test_acc, best_test_loss = 0, float('inf')
    approx_banzhaf_values = defaultdict(float)

    no_improvement_count = 0
    for epoch in range(args.epochs):
        local_weights, local_losses = [], []

        model.train()
        if isBanzhaf:
            gradient = {k: v.to(device) for k, v in test_gradient(model, test_dataset).items()}
        
        m = max(int(args.frac * len(clients)), 1)
        idxs_users = np.random.choice(clients, m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss = local_model.update_weights(model=copy.deepcopy(model), global_round=epoch)
            # Ensure weights are on GPU
            w = {k: v.to(device) for k, v in w.items()}
            local_weights.append(w)
            local_losses.append(loss)

            if isBanzhaf:
                delta_weights = {k: (w[k] - global_weights[k].to(device)) for k in w.keys()}
                b_value = sum(-torch.dot(gradient[k].flatten(), delta_weights[k].flatten()) for k in gradient.keys())
                approx_banzhaf_values[idx] += b_value.item()

        global_weights = average_weights(local_weights)
        model.load_state_dict(global_weights)

        test_acc, test_loss = test_inference(args, model, test_dataset)
        if test_acc > best_test_acc * 1.01 or test_loss < best_test_loss * 0.99:
            best_test_acc, best_test_loss = test_acc, test_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count > 5:
                print(f'Convergence Reached At Round {epoch + 1}')
                break

    return model, approx_banzhaf_values

def train_subset_model(subset, args, train_dataset, test_dataset, user_groups, results_dict):
    # Each process should initialize its own CUDA context
    device = get_device()
    if device.type == 'cuda':
        torch.cuda.set_device(device.index)
    model = initialize_model(args, device)
    model, _ = train_global_model(args, model, train_dataset, test_dataset, user_groups, device, clients=list(subset))
    acc, _ = test_inference(args, model, test_dataset)
    # Store the result in the shared dictionary
    results_dict[subset] = acc

def calculate_shapley_banzhaf(args, results, num_users):
    shapley_values = np.zeros(num_users)
    banzhaf_values = np.zeros(num_users)
    total_permutations = math.factorial(num_users)
    all_clients = list(range(num_users))

    for client in range(num_users):
        for r in range(num_users):
            subsets = list(itertools.combinations([c for c in all_clients if c != client], r))
            for subset in subsets:
                subset = tuple(sorted(subset))
                subset_with_client = tuple(sorted(subset + (client,)))
                result_without = results.get(subset, 0)
                result_with = results.get(subset_with_client, 0)
                marginal_contrib = result_with - result_without
                weight = math.factorial(len(subset)) * math.factorial(num_users - len(subset) - 1) / total_permutations
                shapley_values[client] += weight * marginal_contrib
                banzhaf_values[client] += marginal_contrib
        banzhaf_values[client] /= (2 ** (num_users - 1))

    return shapley_values, banzhaf_values

def run_process(subset, args, train_dataset, test_dataset, user_groups, results_dict, progress_queue):
    device = get_device()
    if device.type == 'cuda':
        torch.cuda.set_device(device.index)
    train_subset_model(subset, args, train_dataset, test_dataset, user_groups, results_dict)
    # Notify the main process that this subset is done
    progress_queue.put(1)

def main():
    start_time = time.time()
    logger = setup_logger('experiment')
    args = args_parser()
    exp_details(args)

    num_cpus = mp.cpu_count()
    num_gpus = torch.cuda.device_count()
    device = get_device()  # Use CUDA device if available

    # Load datasets once in the main process
    train_dataset, test_dataset, user_groups, _, _, _ = get_dataset(args)

    num_users = args.num_users
    all_clients = list(range(num_users))
    all_subsets = []

    for r in range(num_users + 1):
        subsets_r = list(itertools.combinations(all_clients, r))
        all_subsets.extend(subsets_r)

    print(f"Training Models For {len(all_subsets)} Subsets")

    manager = mp.Manager()
    results_dict = manager.dict()

    # Create a queue to track progress
    progress_queue = mp.Queue()
    total_subsets = len(all_subsets)

    # Start processes
    processes = []
    max_processes = num_gpus if num_gpus > 0 else 1
    semaphore = mp.Semaphore(max_processes)

    for subset in all_subsets:
        semaphore.acquire()
        p = mp.Process(target=run_process, args=(
            subset, args, train_dataset, test_dataset, user_groups, results_dict, progress_queue))
        p.start()
        processes.append((p, semaphore))

    # Progress bar
    with tqdm(total=total_subsets, desc="Processing Subsets") as pbar:
        completed_subsets = 0
        while completed_subsets < total_subsets:
            # Wait for a progress update
            progress_queue.get()
            completed_subsets += 1
            pbar.update(1)

    # Join processes
    for p, sem in processes:
        p[0].join()
        sem.release()

    # Convert results to standard dict
    results = dict(results_dict)

    # Calculate Shapley and Banzhaf values
    shapley_values, banzhaf_values = calculate_shapley_banzhaf(args, results, num_users)

    # Train global model with all clients
    if device.type == 'cuda':
        torch.cuda.set_device(device.index)
    global_model = initialize_model(args, device)
    global_model.train()
    clients = list(range(num_users))
    global_model, approx_banzhaf_values = train_global_model(args, global_model, train_dataset, test_dataset, user_groups, device, clients=clients, isBanzhaf=True)
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    # Logging
    logger.info(f'Number Of Clients: {args.num_users}, Client Selection Fraction: {args.frac}, Local Epochs: {args.local_ep}')
    logger.info(f'Dataset: {args.dataset}, Test Accuracy Of Global Model: {100*test_acc}%')
    logger.info(f'Shapley Values: {dict(enumerate(shapley_values))}')
    logger.info(f'Banzhaf Values: {dict(enumerate(banzhaf_values))}')
    logger.info(f'Approximate Banzhaf Values: {approx_banzhaf_values}')
    logger.info(f'Pearson Correlation: {pearsonr(shapley_values, banzhaf_values)}')
    logger.info(f'Spearman Correlation: {spearmanr(shapley_values, banzhaf_values)}')
    logger.info(f'Total Run Time: {time.time() - start_time}')

if __name__ == '__main__':
    mp.set_start_method('spawn')  # Necessary for CUDA with multiprocessing
    main()
