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

    n = 2*args.epochs if isBanzhaf else args.epochs
    for epoch in tqdm(range(n), desc=f"Global Training For Subset {clients}"):
        local_weights, local_losses = [], []

        model.train()
        if isBanzhaf:
            gradient = test_gradient(args, model, valid_dataset)

        m = max(int(args.frac * len(clients)), 1)
        idxs_users = np.random.choice(clients, m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss = local_model.update_weights(model=copy.deepcopy(model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(loss)

            # compute banzhaf value estimate
            if isBanzhaf:
                delta_t[epoch][idx] = {key: (global_weights[key] - w[key]).to(device) for key in w.keys()}

        if isBanzhaf:
            G_t = compute_G_t(delta_t[epoch], global_weights.keys())
            for idx in idxs_users:
                G_t_minus_i = compute_G_minus_i_t(delta_t[epoch], global_weights.keys(), idx)
                if epoch > 0:
                    for key in global_weights.keys():
                        delta_g[idx][key] += G_t_minus_i[key] - G_t[key]
                approx_banzhaf_values_hessian[idx] += compute_bv_hvp(args, model, test_dataset, gradient, delta_t[epoch][idx], delta_g[idx])
                approx_banzhaf_values_simple[idx] += compute_bv_simple(args, gradient, delta_t[epoch][idx])

        # update global weights and model
        global_weights = average_weights(local_weights)
        model.load_state_dict(global_weights)

        test_acc, test_loss = test_inference(model, test_dataset)
        if test_acc > best_test_acc * 1.01 or test_loss < best_test_loss * 0.99:
            best_test_acc = test_acc
            best_test_loss = test_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count > 5:
                print(f'Convergence Reached At Round {epoch + 1}')
                break
        print(f'Best Test Accuracy: {best_test_acc}, Best Test Loss: {best_test_loss}')

    clients_str = "_".join(str(client) for client in clients)
    torch.save(model.state_dict(), f"{clients_str}_epoch_{epoch + 1}_{args.dataset}_{args.setting}.pth")
    return model, approx_banzhaf_values_simple, approx_banzhaf_values_hessian


def train_subset(subset, args, train_dataset, valid_dataset, test_dataset, user_groups):
    device = get_device()
    global_model = initialize_model(args)
    global_model.to(device)
    global_model.train()

    subset_key = tuple(sorted(subset))
    print(f"Training Model For Subset {subset_key}")
    model, _, _ = train_global_model(args, global_model, train_dataset, valid_dataset, test_dataset, user_groups, device, subset)
    accuracy, loss = test_inference(model, test_dataset)
    torch.cuda.empty_cache()
    return (subset_key, loss)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    start_time = time.time()
    args = args_parser()
    logger = setup_logger(f'benchmark_{args.dataset}_{args.setting}_{args.num_users}_{args.local_bs}')
    print(args)

    device = get_device()
    train_dataset, valid_dataset, test_dataset, user_groups, actual_bad_clients = get_dataset(args)

    shapley_values, banzhaf_values = defaultdict(float), defaultdict(float)
    all_subsets = list(itertools.chain.from_iterable(itertools.combinations(range(args.num_users), r) for r in range(args.num_users + 1)))


    pool = multiprocessing.Pool(processes=args.processes)
    train_subset_partial = partial(train_subset, args=args, train_dataset=train_dataset, valid_dataset=valid_dataset, test_dataset=test_dataset, user_groups=user_groups)
    results_list = pool.map(train_subset_partial, all_subsets)
    pool.close()
    pool.join()
    results = dict(results_list)

    for client in range(args.num_users):
        for r in range(args.num_users):
            for subset in itertools.combinations([c for c in range(args.num_users) if c != client], r):
                subset_key = tuple(sorted(subset))
                subset_with_client_key = tuple(sorted(subset + (client,)))
                marginal_contribution = results[subset_key] - results[subset_with_client_key]
                shapley_values[client] += ((math.factorial(len(subset)) * math.factorial(args.num_users - len(subset) - 1)) / math.factorial(args.num_users)) * marginal_contribution
                banzhaf_values[client] += marginal_contribution / len(all_subsets)

    global_model = initialize_model(args)
    global_model.to(device)
    global_model.train()
    clients = [c for c in range(args.num_users)]
    global_model, approx_banzhaf_values_simple, approx_banzhaf_values_hessian = train_global_model(args, global_model, train_dataset, valid_dataset, test_dataset, user_groups, device, clients=clients, isBanzhaf=True)
    test_acc, test_loss = test_inference(global_model, test_dataset)
    
    predicted_bad_client_simple = identify_bad_idxs(approx_banzhaf_values_simple)
    bad_client_accuracy_simple = measure_accuracy(actual_bad_clients, predicted_bad_client_simple)

    predicted_bad_client_hvp = identify_bad_idxs(approx_banzhaf_values_hessian)
    bad_client_accuracy_hvp = measure_accuracy(actual_bad_clients, predicted_bad_client_hvp)

    # remove any clients that are not in approx_banzhaf_values and are not in shapley_values and banzhaf_values 
    shared_clients = set(shapley_values.keys()) & set(banzhaf_values.keys()) & set(approx_banzhaf_values_simple.keys())
    shapley_values = [shapley_values[client] for client in shared_clients]
    banzhaf_values = [banzhaf_values[client] for client in shared_clients]
    approx_banzhaf_values_simple = [approx_banzhaf_values_simple[client] for client in shared_clients]
    approx_banzhaf_values_hessian = [approx_banzhaf_values_hessian[client] for client in shared_clients]
    print(shapley_values)
    print(banzhaf_values)
    print(approx_banzhaf_values_simple)
    print(approx_banzhaf_values_hessian)

    # log results
    if args.setting == 0:
        setting_str = "IID"
    elif args.setting == 1:
        setting_str = f"{len(actual_bad_clients)} Bad Clients" + f" with {args.num_categories_per_client} Categories Per Bad Client"
    elif args.setting == 2:
        setting_str = f"{len(actual_bad_clients)} Bad Clients" + f" with {100*args.mislabel_proportion}% Mislabeled Samples Per Bad Client"
    elif args.setting == 3:
        setting_str = f"{len(actual_bad_clients)} Bad Clients" + f" with {100*args.noise_proportion}% Noisy Samples Per Bad Client"
    logger.info(f'Number Of Clients: {args.num_users}, Client Selection Fraction: {args.frac}, Local Epochs: {args.local_ep}, Batch Size: {args.local_bs}')
    logger.info(f'Dataset: {args.dataset}, Setting: {setting_str}, Number Of Rounds: {args.epochs}')
    logger.info(f'Test Accuracy Of Global Model: {100*test_acc}%')
    logger.info(f'Shapley Values: {shapley_values}')
    logger.info(f'Banzhaf Values: {banzhaf_values}')
    logger.info(f'Approximate Banzhaf Values Simple: {approx_banzhaf_values_simple}')
    logger.info(f'Approximate Banzhaf Values Hessian: {approx_banzhaf_values_hessian}')
    logger.info(f'Pearson Correlation Between Shapley And Banzhaf Values: {pearsonr(shapley_values, banzhaf_values)}')
    logger.info(f'Pearson Correlation Between Shapley And Approximate Banzhaf Values Simple: {pearsonr(shapley_values, approx_banzhaf_values_simple)}')
    logger.info(f'Pearson Correlation Between Shapley And Approximate Banzhaf Values Hessian: {pearsonr(shapley_values, approx_banzhaf_values_hessian)}')
    logger.info(f'Pearson Correlation Between Banzhaf And Approximate Banzhaf Values Simple: {pearsonr(banzhaf_values, approx_banzhaf_values_simple)}')
    logger.info(f'Pearson Correlation Between Banzhaf And Approximate Banzhaf Values Hessian: {pearsonr(banzhaf_values, approx_banzhaf_values_hessian)}')
    logger.info(f'Actual Bad Clients: {actual_bad_clients}')
    logger.info(f'Bad Client Accuracy Simple: {bad_client_accuracy_simple}')
    logger.info(f'Bad Client Accuracy Hessian: {bad_client_accuracy_hvp}')
    logger.info(f'Average Difference Between Banzhaf Values Simple And Hessian: {np.mean(np.abs(np.array(approx_banzhaf_values_simple) - np.array(approx_banzhaf_values_hessian)))}')
    logger.info(f'Total Run Time: {time.time()-start_time}')