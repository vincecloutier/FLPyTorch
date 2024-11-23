import itertools
from math import factorial as fact
from collections import defaultdict
import copy
import time
import torch
import numpy as np
from tqdm import tqdm
from options import args_parser
from update import LocalUpdate, test_inference, test_gradient
from estimation import compute_bv_simple, compute_bv_hvp, compute_G_t, compute_G_minus_i_t
from utils import get_dataset, average_weights, setup_logger, get_device, identify_bad_idxs, measure_accuracy, initialize_model
import multiprocessing
from scipy.stats import pearsonr
from functools import partial

def train_global_model(args, model, train_dataset, valid_dataset, test_dataset, user_groups, device, clients=None, isBanzhaf=False):
    if clients is None or len(clients) == 0:
        return model, defaultdict(float), defaultdict(float)
    global_weights = model.state_dict()
    best_test_acc, best_test_loss = 0, float('inf')
    approx_banzhaf_values_hessian = defaultdict(float)
    approx_banzhaf_values_simple = defaultdict(float)
    if isBanzhaf:
        delta_t = defaultdict(dict)
        delta_g = defaultdict(lambda: {key: torch.zeros_like(global_weights[key]) for key in global_weights.keys()})

    no_improvement_count = 0
    for epoch in tqdm(range(args.epochs), desc=f"Global Training For Subset {clients}"):
        local_weights, local_losses = [], []

        model.train()
        if isBanzhaf:
            gradient = test_gradient(args, model, valid_dataset, device)

        m = max(int(args.frac * len(clients)), 1)
        idxs_users = np.random.choice(clients, m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], device=device)
            w, loss = local_model.update_weights(model=copy.deepcopy(model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(loss)

            if isBanzhaf:
                delta_t[epoch][idx] = {key: (global_weights[key] - w[key]).to(device) for key in w.keys()}

        if isBanzhaf:
            G_t = compute_G_t(delta_t[epoch], global_weights.keys(), device)
            for idx in idxs_users:
                G_t_minus_i = compute_G_minus_i_t(delta_t[epoch], global_weights.keys(), idx, device)
                if epoch > 0:
                    for key in global_weights.keys():
                        delta_g[idx][key] += G_t_minus_i[key] - G_t[key]
                approx_banzhaf_values_hessian[idx] += compute_bv_hvp(args, model, test_dataset, gradient, delta_t[epoch][idx], delta_g[idx], device)
                approx_banzhaf_values_simple[idx] += compute_bv_simple(args, gradient, delta_t[epoch][idx], device)

        global_weights = average_weights(local_weights, device)
        model.load_state_dict(global_weights)

        test_acc, test_loss = test_inference(model, test_dataset, device)
        if test_acc > best_test_acc * 1.01 or test_loss < best_test_loss * 0.99:
            best_test_acc = test_acc
            best_test_loss = test_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count > 3:
                break

    return model, approx_banzhaf_values_simple, approx_banzhaf_values_hessian

def worker(gpu_id, task_queue, args, train_dataset, valid_dataset, test_dataset, user_groups, results_list):
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    global_model = initialize_model(args)
    global_model.to(device)
    global_model.train()

    while True:
        try:
            subset = task_queue.get_nowait()
        except:
            break  # No more tasks

        subset_key = tuple(sorted(subset))
        print(f"GPU {gpu_id} training subset {subset_key}")

        isBanzhaf = subset_key == (0, 1, 2, 3, 4)  # Adjust condition as needed

        model, abv_simple, abv_hessian = train_global_model(
            args, copy.deepcopy(global_model), train_dataset, valid_dataset, test_dataset,
            user_groups, device, clients=subset, isBanzhaf=isBanzhaf
        )
        accuracy, loss = test_inference(model, test_dataset, device)
        results_list.append((subset_key, loss, accuracy, abv_simple, abv_hessian))
        torch.cuda.empty_cache()

def main():
    multiprocessing.set_start_method('spawn') 
    start_time = time.time()
    args = args_parser()
    logger = setup_logger(f'benchmark_{args.dataset}_{args.setting}')
    print(args)

    n_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {n_gpus}")

    processes = min(args.processes if args.processes > 0 else multiprocessing.cpu_count(), n_gpus)
    print(f"Number of GPU processes: {processes}")

    train_dataset, valid_dataset, test_dataset, user_groups, actual_bad_clients = get_dataset(args)

    all_subsets = list(itertools.chain.from_iterable(
        itertools.combinations(range(args.num_users), r) for r in range(args.num_users, -1, -1)
    ))

    task_queue = multiprocessing.Queue()
    for subset in all_subsets:
        task_queue.put(subset)

    manager = multiprocessing.Manager()
    results_list = manager.list()

    workers = []
    for gpu_id in range(n_gpus):
        p = multiprocessing.Process(target=worker, args=(
            gpu_id, task_queue, args, train_dataset, valid_dataset, test_dataset, user_groups, results_list
        ))
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    # Aggregate results
    results = {}
    for subset_key, loss, accuracy, abv_simple, abv_hessian in results_list:
        results[subset_key] = (loss, accuracy, abv_simple, abv_hessian)

    # Compute Shapley and Banzhaf values
    shapley_values, banzhaf_values = defaultdict(float), defaultdict(float)
    for client in range(args.num_users):
        for r in range(args.num_users):
            for subset in itertools.combinations([c for c in range(args.num_users) if c != client], r):
                subset_key = tuple(sorted(subset))
                subset_with_client_key = tuple(sorted(subset + (client,)))
                if subset_key in results and subset_with_client_key in results:
                    mc = results[subset_key][0] - results[subset_with_client_key][0]
                    shapley_values[client] += ((fact(len(subset)) * fact(args.num_users - len(subset) - 1)) / fact(args.num_users)) * mc
                    banzhaf_values[client] += mc / len(all_subsets)

    longest_client_key = max(results.keys(), key=len)
    test_loss, test_acc, abv_simple, abv_hessian = results[longest_client_key]

    identified_bad_clients_simple = identify_bad_idxs(abv_simple)
    identified_bad_clients_hessian = identify_bad_idxs(abv_hessian)
    bad_client_accuracy_simple = measure_accuracy(actual_bad_clients, identified_bad_clients_simple)
    bad_client_accuracy_hessian = measure_accuracy(actual_bad_clients, identified_bad_clients_hessian)

    # remove clients not present in all metrics
    shared_clients = set(shapley_values.keys()) & set(banzhaf_values.keys()) & set(abv_simple.keys()) & set(abv_hessian.keys())
    sv = [shapley_values[client] for client in shared_clients]
    bv = [banzhaf_values[client] for client in shared_clients]
    abv_simple = [abv_simple[client] for client in shared_clients]
    abv_hessian = [abv_hessian[client] for client in shared_clients]

    # Log results
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
    logger.info(f'Approximate Banzhaf Values Simple: {abv_simple}')
    logger.info(f'Approximate Banzhaf Values Hessian: {abv_hessian}')
    logger.info(f'Pearson Correlation Between Shapley And Banzhaf Values: {pearsonr(sv, bv)}')
    logger.info(f'Pearson Correlation Between Shapley And Approximate Banzhaf Values Simple: {pearsonr(sv, abv_simple)}')
    logger.info(f'Pearson Correlation Between Shapley And Approximate Banzhaf Values Hessian: {pearsonr(sv, abv_hessian)}')
    logger.info(f'Pearson Correlation Between Banzhaf And Approximate Banzhaf Values Simple: {pearsonr(bv, abv_simple)}')
    logger.info(f'Pearson Correlation Between Banzhaf And Approximate Banzhaf Values Hessian: {pearsonr(bv, abv_hessian)}')
    logger.info(f'Actual Bad Clients: {actual_bad_clients}')
    logger.info(f'Identified Bad Clients Simple: {identified_bad_clients_simple}')
    logger.info(f'Identified Bad Clients Hessian: {identified_bad_clients_hessian}')
    logger.info(f'Bad Client Accuracy Simple: {bad_client_accuracy_simple}')
    logger.info(f'Bad Client Accuracy Hessian: {bad_client_accuracy_hessian}')
    logger.info(f'Total Run Time: {time.time() - start_time}')

if __name__ == '__main__':
    main()
