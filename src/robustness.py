import copy
import time
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from options import args_parser
from update import LocalUpdate, test_inference, gradient
from utils import get_dataset, average_weights, setup_logger, get_device, initialize_model
from valuation.banzhaf import compute_abv, compute_G_t, compute_G_minus_i_t
from valuation.influence import compute_influence
from valuation.shapley import compute_shapley
import warnings
import multiprocessing
from functools import partial
warnings.filterwarnings("ignore", category=UserWarning)
import os


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def train_client(idx, args, global_weights, train_dataset, user_groups, epoch, device):
    model = initialize_model(args)
    model.load_state_dict(global_weights)
    model.to(device)
    model.train()

    local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
    w, _ = local_model.update_weights(model=model, global_round=epoch)
    delta = {key: (global_weights[key] - w[key]).to(device) for key in global_weights.keys()}

    del local_model, model
    torch.cuda.empty_cache()

    return idx, w, delta


def train_global_model(args, model, train_dataset, valid_dataset, test_dataset, user_groups, device):
    global_weights = model.state_dict()
    abv_simple, abv_hessian, shapley_values, influence_values = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
    delta_t, delta_g = defaultdict(dict), defaultdict(lambda: {key: torch.zeros_like(global_weights[key]) for key in global_weights.keys()})

    runtimes = {'abvs': 0, 'abvh': 0, 'sv': 0, 'if': 0}
    
    for epoch in tqdm(range(args.epochs), desc="Training Epochs"):
        local_weights = []
        local_weights_dict = defaultdict(dict)

        start_time = time.time()
        grad = gradient(args, model, valid_dataset)
        runtimes['abvs'] += time.time() - start_time
        runtimes['abvh'] += time.time() - start_time

        # no randomization
        idxs_users = range(args.num_users)
        
        train_client_partial = partial(train_client, args=args, global_weights=copy.deepcopy(global_weights), train_dataset=train_dataset, user_groups=user_groups, epoch=epoch, device=device)
        with multiprocessing.Pool(processes=args.processes) as pool:
            results = pool.map(train_client_partial, idxs_users)
        pool.close()
        pool.join()

        for idx, w, delta in results:
            local_weights.append(copy.deepcopy(w))
            local_weights_dict[idx] = copy.deepcopy(w)
            delta_t[epoch][idx] = delta

        # compute shapley values
        start_time = time.time()
        shapley_updates = compute_shapley(args, global_weights, local_weights_dict, test_dataset)
        for k, v in shapley_updates.items():
            shapley_values[k] += v  
        runtimes['sv'] += (time.time() - start_time) * args.shapley_processes

        global_weights = average_weights(local_weights)
       
        # compute banzhaf values
        start_time = time.time()
        G_t = compute_G_t(delta_t[epoch], global_weights.keys())
        for idx in idxs_users:
            G_t_minus_i = compute_G_minus_i_t(delta_t[epoch], global_weights.keys(), idx)
            if epoch > 0:
                for key in global_weights.keys():
                    delta_g[idx][key] += G_t_minus_i[key] - G_t[key]
            t_time = time.time() - start_time
            runtimes['abvh'] += t_time
            runtimes['abvs'] += t_time
            start_time = time.time()
            abv_hessian[idx] += compute_abv(args, model, train_dataset, user_groups[idx], grad, delta_t[epoch][idx], delta_g[idx], is_hessian=True)
            runtimes['abvh'] += time.time() - start_time
            start_time = time.time()
            abv_simple[idx] += compute_abv(args, model, train_dataset, user_groups[idx], grad, delta_t[epoch][idx], delta_g[idx], is_hessian=False)
            runtimes['abvs'] += time.time() - start_time

        model.load_state_dict(global_weights)
    
    # compute influence values
    start_time = time.time()
    influence_values = compute_influence(args, global_weights, train_dataset, test_dataset, user_groups)
    runtimes['if'] += time.time() - start_time

    return model, abv_simple, abv_hessian, shapley_values, influence_values, runtimes

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    args = args_parser()
    logger = setup_logger(f'robustness_{args.dataset}_{args.setting}')
    print(args)

    device = get_device()
    train_dataset, valid_dataset, test_dataset, user_groups, actual_bad_clients = get_dataset(args)
    
    global_model = initialize_model(args)
    global_model.to(device)
    global_params = global_model.state_dict()


    for i in range(3):
        global_model.load_state_dict(global_params)
        logger.info(f'Run {i}')

        global_model, abv_simple, abv_hessian, shapley_values, influence_values, runtimes = train_global_model(args, global_model, train_dataset, valid_dataset, test_dataset, user_groups, device)
        test_acc, test_loss = test_inference(global_model, test_dataset)

        # log results   
        if args.setting == 0:
            setting_str = "IID"
        elif args.setting == 1:
            setting_str = f"Non IID with {len(actual_bad_clients)} Bad Clients and {args.num_categories_per_client} Categories Per Bad Client"
        elif args.setting == 2:
            setting_str = f"Mislabeled with {len(actual_bad_clients)} Bad Clients and {100 * args.badsample_prop}% Bad Samples Per Bad Client"
        elif args.setting == 3:
            setting_str = f"Noisy with {len(actual_bad_clients)} Bad Clients and {100 * args.badsample_prop}% Bad Samples Per Bad Client"
        
        logger.info(f'Number Of Clients: {args.num_users}, Client Selection Fraction: {args.frac}, Local Epochs: {args.local_ep}')
        logger.info(f'Batch Size: {args.local_bs}, Learning Rate: {args.lr}')
        logger.info(f'Dataset: {args.dataset}, Setting: {setting_str}, Number Of Rounds: {args.epochs}')
        logger.info(f'Test Accuracy: {100*test_acc}%, Test Loss: {test_loss}')
        logger.info(f'Banzhaf Values Simple: {abv_simple}')
        logger.info(f'Banzhaf Values Hessian: {abv_hessian}')
        logger.info(f'Shapley Values: {shapley_values}')
        logger.info(f'Influence Function Values: {influence_values}')
        logger.info(f'Actual Bad Clients: {actual_bad_clients}')
        logger.info(f'Runtimes: {runtimes}')