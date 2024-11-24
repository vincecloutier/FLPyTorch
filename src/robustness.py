import copy
import time
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from options import args_parser
from update import LocalUpdate, test_inference, test_gradient, conjugate_gradient
from utils import get_dataset, average_weights, setup_logger, get_device, identify_bad_idxs, measure_accuracy, initialize_model
from estimation import compute_bv_hvp, compute_bv_simple, compute_G_t, compute_G_minus_i_t
import warnings
import multiprocessing
from functools import partial
warnings.filterwarnings("ignore", category=UserWarning)
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import random

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

def compute_monte_carlo_shapley(args, global_weights, train_dataset, user_groups, device, num_samples=10):
    """Compute Monte Carlo Shapley values for clients."""
    num_clients = args.num_users
    shapley_values = defaultdict(float)
    
    clients = list(range(num_clients))
    
    for sample in tqdm(range(num_samples), desc="Computing Shapley Values"):
        # generate a random permutation of clients
        random_permutation = copy.deepcopy(clients)
        random.shuffle(random_permutation)
        
        current_weights = copy.deepcopy(global_weights)
        current_model = initialize_model(args)
        current_model.load_state_dict(current_weights)
        current_model.to(device)
        current_model.train()
        
        # initial performance
        initial_perf = test_inference(current_model, test_dataset)[0]  # assuming higher is better
        
        for i, client_idx in enumerate(random_permutation):
            # train the model with the current client
            local_update = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[client_idx])
            w, _ = local_update.update_weights(model=current_model, global_round=sample)
            current_weights = average_weights([current_weights, w])
            current_model.load_state_dict(current_weights)
            
            # performance after adding the client
            new_perf = test_inference(current_model, test_dataset)[0]
            
            # marginal contribution
            marginal_contribution = new_perf - initial_perf
            shapley_values[client_idx] += marginal_contribution
            
            # update initial performance for next client in permutation
            initial_perf = new_perf
        
        del current_model
        torch.cuda.empty_cache()
    
    # Average the contributions
    for client_idx in shapley_values:
        shapley_values[client_idx] /= num_samples
    
    return shapley_values

def compute_influence_functions(args, model, train_dataset, user_groups, device):
    """Compute Influence Functions for clients."""
    influence_values = defaultdict(float)
    model.eval()
    
    # compute the gradient of the test loss w.r.t. model parameters
    test_loss_grad = test_gradient(args, model, test_dataset)
    
    # solve Hx = grad_test_loss to get x = H^{-1} grad_test_loss
    x = conjugate_gradient(model, train_dataset, test_loss_grad, num_iterations=20, tol=1e-4)
    
    # split x back into parameter tensors
    x_dict = {}
    idx = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            numel = param.numel()
            x_dict[name] = x[idx:idx+numel].view_as(param).clone().detach()
            idx += numel
    
    # iterate over each client to compute their influence
    for client_idx in tqdm(range(args.num_users), desc="Computing Influence Functions"):
        client_data = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[client_idx]).get_data()
        client_grad = test_gradient(args, model, client_data)
        client_grad_flat = torch.cat([g.contiguous().view(-1) for g in client_grad])
        influence = -torch.dot(client_grad_flat, x).item()        
        influence_values[client_idx] = influence
    
    return influence_values

def train_global_model(args, model, train_dataset, test_dataset, user_groups, device):
    global_weights = model.state_dict()
    best_test_acc, best_test_loss = 0, float('inf')
    approx_banzhaf_values_simple = defaultdict(float)
    approx_banzhaf_values_hvp = defaultdict(float)
    shapley_values = defaultdict(float)
    influence_values = defaultdict(float)
    
    delta_t = defaultdict(dict)
    delta_g = defaultdict(lambda: {key: torch.zeros_like(global_weights[key]) for key in global_weights.keys()})

    no_improvement_count = 0
    for epoch in tqdm(range(args.epochs), desc="Training Epochs"):
        local_weights = []

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
            delta_t[epoch][idx] = delta

        global_weights = average_weights(local_weights)
        model.load_state_dict(global_weights)

        # Compute Banzhaf values
        G_t = compute_G_t(delta_t[epoch], global_weights.keys())
        for idx in idxs_users:
            G_t_minus_i = compute_G_minus_i_t(delta_t[epoch], global_weights.keys(), idx)
            if epoch > 0:
                for key in global_weights.keys():
                    delta_g[idx][key] += G_t_minus_i[key] - G_t[key]
            approx_banzhaf_values_hvp[idx] += compute_bv_hvp(args, model, test_dataset, gradient, delta_t[epoch][idx], delta_g[idx])
            approx_banzhaf_values_simple[idx] += compute_bv_simple(args, gradient, delta_t[epoch][idx])

        # compute mc shapley values and influence functions periodically
        shapley = compute_monte_carlo_shapley(args, global_weights, train_dataset, user_groups, device, num_samples=50)
        influence = compute_influence_functions(args, model, train_dataset, user_groups, device)
        for k, v in shapley.items():
            shapley_values[k] += v  
        for k, v in influence.items():
            influence_values[k] += v 

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

        print(f'Epoch {epoch+1}/{args.epochs} - Test Accuracy: {test_acc}, Test Loss: {test_loss}')
        print(torch.cuda.memory_summary(device=device))

    # average shapley and influence values over the number of sampling points
    if shapley_values:
        for k in shapley_values:
            shapley_values[k] /= (args.epochs // 5)
    if influence_values:
        for k in influence_values:
            influence_values[k] /= (args.epochs // 5)

    return model, approx_banzhaf_values, shapley_values, influence_values

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    start_time = time.time()
    args = args_parser()
    logger = setup_logger(f'robustness_{args.dataset}_{args.setting}')
    print(args)

    device = get_device()
    train_dataset, valid_dataset, test_dataset, user_groups, actual_bad_clients = get_dataset(args)
    
    torch.cuda.set_per_process_memory_fraction(0.25, device)

    # train the global model
    global_model = initialize_model(args)
    global_model.to(device)
    global_model.train()
    global_model, approx_banzhaf_values, shapley_values, influence_values = train_global_model(args, global_model, train_dataset, test_dataset, user_groups, device)
    test_acc, test_loss = test_inference(global_model, test_dataset)

    # predict bad clients and measure accuracy
    predicted_bad_clients = identify_bad_idxs(approx_banzhaf_values)
    bad_client_accuracy = measure_accuracy(actual_bad_clients, predicted_bad_clients)

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
    logger.info(f'Banzhaf Values: {approx_banzhaf_values}')
    logger.info(f'Shapley Values: {shapley_values}')
    logger.info(f'Influence Function Values: {influence_values}')
    logger.info(f'Actual Bad Clients: {actual_bad_clients}')
    logger.info(f'Predicted Bad Clients: {predicted_bad_clients}')
    logger.info(f'Bad Client Accuracy: {bad_client_accuracy}')
    logger.info(f'Total Run Time: {time.time()-start_time}')
