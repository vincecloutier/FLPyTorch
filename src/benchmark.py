import itertools
from math import factorial as fact
from collections import defaultdict
import time
import torch
import numpy as np
from options import args_parser
from update import inference, gradient
from torch.utils.data import DataLoader
from estimation import compute_bv_simple, compute_bv_hvp, compute_G_t, compute_G_minus_i_t
from utils import average_weights, setup_logger, get_device, identify_bad_idxs, measure_accuracy, initialize_model
from sampling import get_dataset, SubsetSplit
from scipy.stats import pearsonr
import copy
from torch import nn
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Initialize a lock for tqdm to prevent overlapping prints
tqdm_lock = threading.Lock()

def train_subset(args, global_weights, train_loaders, valid_dataset, test_dataset, clients, device, position):
    if len(clients) == 0:
        return (), 0, 0, {}, {}
    
    global_model = initialize_model(args)
    global_model.to(device)
    global_model.load_state_dict(global_weights)
    global_model.train()

    subset_key = tuple(sorted(clients))
    isBanzhaf = subset_key == (0, 1, 2, 3, 4) 
    best_test_acc, best_test_loss = 0, float('inf')
    approx_banzhaf_values_hessian = defaultdict(float)
    approx_banzhaf_values_simple = defaultdict(float)
    delta_t = defaultdict(dict) if isBanzhaf else None
    delta_g = defaultdict(lambda: {key: torch.zeros_like(global_weights[key]) for key in global_weights.keys()}) if isBanzhaf else None
    no_improvement_count = 0

    # Initialize a tqdm progress bar for this subset
    with tqdm(total=args.epochs, desc=f"Subset: {subset_key}", position=position, leave=True) as pbar:
        for epoch in range(args.epochs):
            local_weights = []

            # sample a fraction of clients
            m = max(int(args.frac * len(subset_key)), 1)
            idxs_users = np.random.choice(subset_key, m, replace=False)

            for idx in idxs_users:
                w = train_client(args, global_weights, train_loaders[idx], device, 1 / (len(idxs_users) * args.local_ep), pbar)
                local_weights.append(w)
                if isBanzhaf:
                    delta_t[epoch][idx] = {key: (global_weights[key] - w[key]).to(device) for key in w.keys()}

            if isBanzhaf:
                grad = gradient(args, global_model, valid_dataset, device)
                G_t = compute_G_t(delta_t[epoch], global_weights.keys())
                for idx in idxs_users:
                    G_t_minus_i = compute_G_minus_i_t(delta_t[epoch], global_weights.keys(), idx)
                    if epoch > 0:
                        for key in global_weights.keys():
                            delta_g[idx][key] += G_t_minus_i[key] - G_t[key]
                    approx_banzhaf_values_hessian[idx] += compute_bv_hvp(args, global_model, test_dataset, grad, delta_t[epoch][idx], delta_g[idx], device)
                    approx_banzhaf_values_simple[idx] += compute_bv_simple(args, grad, delta_t[epoch][idx])

            # average the local weights to update global weights
            global_weights = average_weights(local_weights)
            global_model.load_state_dict(global_weights)

            # evaluate the global model
            test_acc, test_loss = inference(args, global_model, test_dataset, device)
            if test_acc > best_test_acc * 1.01 or test_loss < best_test_loss * 0.99:
                best_test_acc = test_acc
                best_test_loss = test_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count > 3:
                    print(f"No improvement for {no_improvement_count} consecutive epochs. Stopping early.")
                    break

            # update the progress bar
            with tqdm_lock:
                pbar.update(1)
    
    torch.cuda.empty_cache()
    return subset_key, best_test_loss, best_test_acc, approx_banzhaf_values_simple, approx_banzhaf_values_hessian


def train_client(args, global_weights, trainloader, device, fractional_update, global_pbar):
    model = initialize_model(args)
    model.to(device)
    model.load_state_dict(global_weights)
    model.train()

    epoch_loss = []

    criterion = nn.CrossEntropyLoss().to(device)

    # set optimizer 
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    for _ in range(args.local_ep):
        batch_loss = []
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            log_probs = model(images)
            loss = criterion(log_probs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # update both local and global progress bars
        with tqdm_lock:
            global_pbar.update(fractional_update)

    return model.state_dict()


if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    logger = setup_logger(f'benchmark_{args.dataset}_{args.setting}')

    # load datasets
    train_dataset, valid_dataset, test_dataset, user_groups, actual_bad_clients = get_dataset(args)

    # initialize global model
    global_model = initialize_model(args)
    global_weights = global_model.state_dict()

    # prepare dataloaders for each client
    train_loaders = {
        user_id: DataLoader(SubsetSplit(train_dataset, indices), batch_size=args.local_bs, shuffle=True, num_workers=args.num_workers)
        for user_id, indices in user_groups.items()
    }

    all_subsets = list(itertools.chain.from_iterable(itertools.combinations(range(args.num_users), r) for r in range(args.num_users, -1, -1)))

    # distribute subsets evenly across devices
    available_gpus = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [torch.device("cpu")]
    subset_device_mapping = {subset: available_gpus[i % len(available_gpus)] for i, subset in enumerate(all_subsets)}

    print(f"Available devices: {available_gpus}")

    position_mapping = {subset: idx % len(all_subsets) for idx, subset in enumerate(all_subsets)}

    # launch train_subset tasks in parallel with controlled concurrency
    results_list = []
    with ThreadPoolExecutor(max_workers=args.processes) as executor:
        # create a future for each subset with its position
        future_to_subset = {
            executor.submit(train_subset, args, global_weights, train_loaders, valid_dataset, test_dataset, subset, subset_device_mapping[subset], position_mapping[subset]): subset
            for subset in all_subsets
        }

        for future in as_completed(future_to_subset):
            subset = future_to_subset[future]
            try:
                result = future.result()
                results_list.append(result)
            except Exception as exc:
                print(f'Subset {subset} generated an exception: {exc}')

    results = {subset_key: (loss, accuracy, abv_simple, abv_hessian) for subset_key, loss, accuracy, abv_simple, abv_hessian in results_list}

    # compute shapley and banzhaf values
    shapley_values, banzhaf_values = defaultdict(float), defaultdict(float)
    for client in range(args.num_users):
        for r in range(args.num_users):
            for subset in itertools.combinations([c for c in range(args.num_users) if c != client], r):
                subset_key = tuple(sorted(subset))
                subset_with_client_key = tuple(sorted(subset + (client,)))
                if subset_key in results and subset_with_client_key in results:
                    mc = results[subset_key][0] - results[subset_with_client_key][0]
                    shapley_values[client] += mc * ((fact(len(subset)) * fact(args.num_users - len(subset) - 1)) / fact(args.num_users))
                    banzhaf_values[client] += mc / len(all_subsets)

    # identify the subset with the longest key (assuming it's the full set)
    longest_client_key = max(results.keys(), key=lambda x: len(x))
    test_loss, test_acc, abv_simple, abv_hessian = results[longest_client_key]

    # identify bad clients using approximate Banzhaf values
    identified_bad_clients_simple = identify_bad_idxs(abv_simple)
    identified_bad_clients_hessian = identify_bad_idxs(abv_hessian)
    bad_client_accuracy_simple = measure_accuracy(actual_bad_clients, identified_bad_clients_simple)
    bad_client_accuracy_hessian = measure_accuracy(actual_bad_clients, identified_bad_clients_hessian)

    # prepare data for correlation
    shared_clients = set(shapley_values.keys()) & set(banzhaf_values.keys()) & set(abv_simple.keys()) & set(abv_hessian.keys())
    sv = [shapley_values[client] for client in shared_clients]
    bv = [banzhaf_values[client] for client in shared_clients]
    abv_simple_list = [abv_simple[client] for client in shared_clients]
    abv_hessian_list = [abv_hessian[client] for client in shared_clients]

    # logging results
    setting_str = {
        0: "IID",
        1: f"Non IID with {len(actual_bad_clients)} Bad Clients and {args.num_categories_per_client} Categories Per Bad Client",
        2: f"Mislabeled with {len(actual_bad_clients)} Bad Clients and {100 * args.badsample_prop}% Bad Samples Per Bad Client",
        3: f"Noisy with {len(actual_bad_clients)} Bad Clients and {100 * args.badsample_prop}% Bad Samples Per Bad Client"
    }.get(args.setting, "Unknown Setting")
    logger.info(f'Number Of Clients: {args.num_users}, Client Selection Fraction: {args.frac}, Local Epochs: {args.local_ep}, Batch Size: {args.local_bs}')
    logger.info(f'Dataset: {args.dataset}, Setting: {setting_str}, Number Of Rounds: {args.epochs}')
    logger.info(f'Test Accuracy Of Global Model: {100 * test_acc}%')
    logger.info(f'Shapley Values: {sv}')
    logger.info(f'Banzhaf Values: {bv}')
    logger.info(f'Approximate Banzhaf Values Simple: {abv_simple_list}')
    logger.info(f'Approximate Banzhaf Values Hessian: {abv_hessian_list}')
    logger.info(f'Pearson Correlation Between Shapley And Banzhaf Values: {pearsonr(sv, bv)}')
    logger.info(f'Pearson Correlation Between Shapley And Approximate Banzhaf Values Simple: {pearsonr(sv, abv_simple_list)}')
    logger.info(f'Pearson Correlation Between Shapley And Approximate Banzhaf Values Hessian: {pearsonr(sv, abv_hessian_list)}')
    logger.info(f'Pearson Correlation Between Banzhaf And Approximate Banzhaf Values Simple: {pearsonr(bv, abv_simple_list)}')
    logger.info(f'Pearson Correlation Between Banzhaf And Approximate Banzhaf Values Hessian: {pearsonr(bv, abv_hessian_list)}')
    logger.info(f'Actual Bad Clients: {actual_bad_clients}')
    logger.info(f'Identified Bad Clients Simple: {identified_bad_clients_simple}')
    logger.info(f'Identified Bad Clients Hessian: {identified_bad_clients_hessian}')
    logger.info(f'Bad Client Accuracy Simple: {bad_client_accuracy_simple}')
    logger.info(f'Bad Client Accuracy Hessian: {bad_client_accuracy_hessian}')
    logger.info(f'Total Run Time: {time.time() - start_time} seconds')