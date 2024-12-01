import copy
import time
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from options import args_parser
from update import LocalUpdate, test_inference, test_gradient
from utils import get_dataset, average_weights, setup_logger, get_device, identify_bad_idxs, measure_accuracy, initialize_model
from valuation.banzhaf import compute_abv, compute_G_t, compute_G_minus_i_t
from valuation.influence import compute_influence_functions
from valuation.shapley import compute_shapley
import warnings
import multiprocessing
from functools import partial
warnings.filterwarnings("ignore", category=UserWarning)
import os


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def train_client(idx, args, global_weights, train_dataset, user_groups, epoch, device):
    torch.cuda.set_device(device)

    model = initialize_model(args)
    model.load_state_dict(global_weights)
    model.to(device)
    model.train()

    local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
    w, _ = local_model.update_weights(model=model, global_round=epoch)
    delta = {key: (w[key] - global_weights[key]).to(device) for key in global_weights.keys()}

    del model
    del local_model
    torch.cuda.empty_cache()

    return idx, w, delta


def train_global_model(args, model, train_dataset, test_dataset, user_groups, device):
    global_weights = model.state_dict()
    best_test_acc, best_test_loss = 0, float('inf')
    abv_simple, abv_hessian, shapley_values, influence_values = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
    runtimes = {'abvs': 0, 'abvh': 0, 'sv': 0, 'if': 0}
    delta_t, delta_g = defaultdict(dict), defaultdict(lambda: {key: torch.zeros_like(global_weights[key]) for key in global_weights.keys()})

    no_improvement_count = 0
    for epoch in tqdm(range(args.epochs), desc="Training Epochs"):
        local_weights = []
        local_weights_dict = defaultdict(dict)

        model.train()
        gradient = test_gradient(args, model, test_dataset)
        
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        train_client_partial = partial(
            train_client, args=args, global_weights=copy.deepcopy(global_weights), train_dataset=train_dataset, user_groups=user_groups, epoch=epoch, device=device
        )

        with multiprocessing.Pool(processes=args.processes) as pool:
            results = pool.map(train_client_partial, idxs_users)
        pool.close()
        pool.join()
        for idx, w, delta in results:
            local_weights.append(copy.deepcopy(w))
            local_weights_dict[idx] = copy.deepcopy(w)
            delta_t[epoch][idx] = delta

        global_weights = average_weights(local_weights)
        model.load_state_dict(global_weights)

        print('computing banzhaf values')
        # compute banzhaf values
        G_t = compute_G_t(delta_t[epoch], global_weights.keys())
        for idx in idxs_users:
            G_t_minus_i = compute_G_minus_i_t(delta_t[epoch], global_weights.keys(), idx)
            if epoch > 0:
                for key in global_weights.keys():
                    delta_g[idx][key] += G_t_minus_i[key] - G_t[key]
            start_time = time.time()
            abv_hessian[idx] += compute_abv(args, model, train_dataset, gradient, delta_t[epoch][idx], delta_g[idx], is_hessian=True)
            runtimes['abvh'] += time.time() - start_time
            start_time = time.time()
            abv_simple[idx] += compute_abv(args, model, train_dataset, gradient, delta_t[epoch][idx], is_hessian=False)
            runtimes['abvs'] += time.time() - start_time

        print('computing shapley values')
        # compute shapley values
        start_time = time.time()
        shapley = compute_shapley(args, global_weights, local_weights_dict, test_dataset)
        runtimes['sv'] += time.time() - start_time
        for k, v in shapley.items():
            shapley_values[k] += v  

        # compute influence values
        # start_time = time.time()
        # influence = compute_influence_functions(args, model, train_dataset, user_groups, device, test_dataset)
        # runtimes['if'] += time.time() - start_time
        # for k, v in influence.items():
        #     influence_values[k] += v 

        test_acc, test_loss = test_inference(model, test_dataset)
        if test_acc > best_test_acc * 1.01 or test_loss < best_test_loss * 0.99:
            best_test_acc = test_acc
            best_test_loss = test_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count > 3:
                print(f'Convergence Reached At Round {epoch + 1}')
                break
        print(shapley_values)
        print(f'Epoch {epoch+1}/{args.epochs} - Test Accuracy: {test_acc}, Test Loss: {test_loss}')
        print(torch.cuda.memory_summary(device=device))

    return model, abv_simple, abv_hessian, shapley_values, influence_values, runtimes

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    # if torch.cuda.is_available():
    #     torch.backends.cudnn.benchmark = True
    #     torch.backends.cudnn.enabled = True

    start_time = time.time()
    args = args_parser()
    logger = setup_logger(f'robustness_{args.dataset}_{args.setting}')
    print(args)

    device = get_device()
    train_dataset, valid_dataset, test_dataset, user_groups, actual_bad_clients = get_dataset(args)
    
    # torch.cuda.set_per_process_memory_fraction(0.25, device)

    # train the global model
    global_model = initialize_model(args)
    global_model.to(device)
    global_model.train()
    global_model, abv_simple, abv_hessian, shapley_values, influence_values, runtimes = train_global_model(args, global_model, train_dataset, test_dataset, user_groups, device)
    test_acc, test_loss = test_inference(global_model, test_dataset)

    shared_clients = set(shapley_values.keys()) & set(influence_values.keys()) & set(abv_simple.keys()) & set(abv_hessian.keys())
    sv = [shapley_values[client] for client in shared_clients]
    iv = [influence_values[client] for client in shared_clients]
    abv_simple = [abv_simple[client] for client in shared_clients]
    abv_hessian = [abv_hessian[client] for client in shared_clients]

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
    logger.info(f'Batch Size: {args.local_bs}, Learning Rate: {args.lr}, Momentum: {args.momentum}')
    logger.info(f'Dataset: {args.dataset}, Setting: {setting_str}, Number Of Rounds: {args.epochs}, Noise STD: {args.noise_std}')
    logger.info(f'Test Accuracy: {100*test_acc}%')
    logger.info(f'Shapley Values: {sv}')
    logger.info(f'Influence Function Values: {iv}')
    logger.info(f'Banzhaf Values Simple: {abv_simple}')
    logger.info(f'Banzhaf Values Hessian: {abv_hessian}')
    logger.info(f'Runtimes: {runtimes}')
    logger.info(f'Total Run Time: {time.time()-start_time}')
