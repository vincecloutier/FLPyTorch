import itertools
from math import factorial as fact
from collections import defaultdict
import copy
import time
import torch
from tqdm import tqdm
from options import args_parser
from update import LocalUpdate, test_inference, gradient
from valuation.banzhaf import compute_abv, compute_G_t, compute_G_minus_i_t
from utils import get_dataset, average_weights, setup_logger, get_device, initialize_model
import multiprocessing
from functools import partial
import numpy as np
import random

def train_global_model(args, model, train_dataset, valid_dataset, test_dataset, user_groups, device, j):
    global_weights = model.state_dict()
    abv_simple, abv_hessian = defaultdict(float), defaultdict(float)
    delta_t, delta_g = defaultdict(dict), defaultdict(lambda: {key: torch.zeros_like(global_weights[key]) for key in global_weights.keys()})

    for epoch in tqdm(range(args.epochs), desc=f"Global Training For Subset {j}"):
        local_weights = []
        grad = gradient(args, model, valid_dataset)
        idxs_users = range(args.num_users)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, _ = local_model.update_weights(model=copy.deepcopy(model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            delta_t[epoch][idx] = {key: (global_weights[key] - w[key]).to(device) for key in w.keys()}
        
        global_weights = average_weights(local_weights)

        G_t = compute_G_t(delta_t[epoch], global_weights.keys())
        for idx in idxs_users:
            G_t_minus_i = compute_G_minus_i_t(delta_t[epoch], global_weights.keys(), idx)
            if epoch > 0:
                for key in global_weights.keys():
                    delta_g[idx][key] += G_t_minus_i[key] - G_t[key]
            abv_hessian[idx] += compute_abv(args, model, train_dataset, user_groups[idx], grad, delta_t[epoch][idx], delta_g[idx], is_hessian=True)
            abv_simple[idx] += compute_abv(args, model, train_dataset, user_groups[idx], grad, delta_t[epoch][idx], delta_g[idx], is_hessian=False)

        model.load_state_dict(global_weights)

        acc, loss = test_inference(model, test_dataset)

        print(f'Run {j}: Epoch {epoch+1}/{args.epochs}, Test Accuracy: {acc}, Test Loss: {loss}')

    return abv_simple, abv_hessian


def train_subset(j, args, train_dataset, valid_dataset, test_dataset, user_groups):
    device = get_device()
    global_model = initialize_model(args)
    global_model.to(device)
    global_model.train()
    print(f"Training Model {j}")
    abv_simple, abv_hessian = train_global_model(args, global_model, train_dataset, valid_dataset, test_dataset, user_groups, device, j)
    return (j, abv_simple, abv_hessian)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    start_time = time.time()
    args = args_parser()
    logger = setup_logger(f'robustness2_{args.dataset}_{args.setting}')
    print(args)

    device = get_device()
    train_dataset, valid_dataset, test_dataset, user_groups, actual_bad_clients = get_dataset(args)

    pool = multiprocessing.Pool(processes=args.processes)
    train_subset_partial = partial(train_subset, args=args, train_dataset=train_dataset, valid_dataset=valid_dataset, test_dataset=test_dataset, user_groups=user_groups)
    results_list = pool.map(train_subset_partial, range(10))
    pool.close()
    pool.join()

    if args.setting == 0:
            setting_str = "IID"
    elif args.setting == 1:
            setting_str = f"Non IID with {len(actual_bad_clients)} Bad Clients and {args.num_categories_per_client} Categories Per Bad Client"
    elif args.setting == 2:
            setting_str = f"Mislabeled with {len(actual_bad_clients)} Bad Clients and {100 * args.badsample_prop}% Bad Samples Per Bad Client"
    elif args.setting == 3:
        setting_str = f"Noisy with {len(actual_bad_clients)} Bad Clients and {100 * args.badsample_prop}% Bad Samples Per Bad Client"
    logger.info(f'Number Of Clients: {args.num_users}, Client Selection Fraction: {args.frac}, Local Epochs: {args.local_ep}, Batch Size: {args.local_bs}')
    logger.info(f'Dataset: {args.dataset}, Setting: {setting_str}, Number Of Rounds: {args.epochs}')

    for j, abv_simple, abv_hessian in results_list:
        logger.info(f'ABV Simple {j}: {abv_simple}')
        logger.info(f'ABV Hessian {j}: {abv_hessian}')

    logger.info(f'Actual Bad Clients: {actual_bad_clients}')
    logger.info(f'Total Run Time: {time.time() - start_time}')