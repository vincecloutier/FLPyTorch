import itertools
from math import factorial as fact
from collections import defaultdict
import copy
import time
import torch
import numpy as np
from tqdm import tqdm
from options import args_parser
from update import LocalUpdate, test_inference, gradient
from valuation.banzhaf import compute_abv, compute_G_t, compute_G_minus_i_t
from utils import get_dataset, average_weights, setup_logger, get_device, identify_bad_idxs, measure_accuracy, initialize_model, EarlyStopping
import multiprocessing
from scipy.stats import pearsonr
from functools import partial

def train_global_model(args, model, train_dataset, valid_dataset, test_dataset, user_groups, device, clients=None, isBanzhaf=False):
    if clients is None or len(clients) == 0:
        return model, defaultdict(float), defaultdict(float)
    global_weights = model.state_dict()
    abv_simple, abv_hessian = defaultdict(float), defaultdict(float)
    if isBanzhaf:
        delta_t, delta_g = defaultdict(dict), defaultdict(lambda: {key: torch.zeros_like(global_weights[key]) for key in global_weights.keys()})

    early_stopping = EarlyStopping(args)

    for epoch in tqdm(range(args.epochs), desc=f"Global Training For Subset {clients}"):
        local_weights = []

        if isBanzhaf:
            grad = gradient(args, model, valid_dataset)

        idxs_users = clients

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, _ = local_model.update_weights(model=copy.deepcopy(model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            if isBanzhaf:
                delta_t[epoch][idx] = {key: (global_weights[key] - w[key]).to(device) for key in w.keys()}
        
        global_weights = average_weights(local_weights)

        if isBanzhaf:
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
        if early_stopping.check(epoch, acc, loss):
            print(f"Convergence Reached At Round {epoch + 1}")
            break

        print(f'Subset {clients} - Epoch {epoch+1}/{args.epochs} - Test Accuracy: {acc}, Test Loss: {loss}')

    return model, abv_simple, abv_hessian


def train_subset(subset, args, train_dataset, valid_dataset, test_dataset, user_groups):
    device = get_device()
    global_model = initialize_model(args)
    global_model.to(device)
    global_model.train()

    subset_key = tuple(sorted(subset))
    print(f"Training Model For Subset {subset_key}")
    if subset_key == (0, 1, 2, 3, 4):
        isBanzhaf = True
    else:
        isBanzhaf = False
    model, abv_simple, abv_hessian = train_global_model(args, global_model, train_dataset, valid_dataset, test_dataset, user_groups, device, subset, isBanzhaf)
    accuracy, loss = test_inference(model, test_dataset)
    del model
    torch.cuda.empty_cache()
    return (subset_key, loss, accuracy, abv_simple, abv_hessian)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    start_time = time.time()
    args = args_parser()
    logger = setup_logger(f'benchmark_{args.dataset}_{args.setting}')
    print(args)

    device = get_device()
    train_dataset, valid_dataset, test_dataset, user_groups, actual_bad_clients = get_dataset(args)

    shapley_values, banzhaf_values = defaultdict(float), defaultdict(float)
    all_subsets = list(itertools.chain.from_iterable(itertools.combinations(range(args.num_users), r) for r in range(args.num_users, -1, -1)))

    pool = multiprocessing.Pool(processes=args.processes)
    train_subset_partial = partial(train_subset, args=args, train_dataset=train_dataset, valid_dataset=valid_dataset, test_dataset=test_dataset, user_groups=user_groups)
    results_list = pool.map(train_subset_partial, all_subsets)
    pool.close()
    pool.join()
    results = {}
    for subset_key, loss, accuracy, abv_simple, abv_hessian in results_list:
        results[subset_key] = (loss, accuracy, abv_simple, abv_hessian)

    for client in range(args.num_users):
        for r in range(args.num_users):
            for subset in itertools.combinations([c for c in range(args.num_users) if c != client], r):
                subset_key = tuple(sorted(subset))
                subset_with_client_key = tuple(sorted(subset + (client,)))
                mc = results[subset_key][0] - results[subset_with_client_key][0]
                shapley_values[client] += ((fact(len(subset)) * fact(args.num_users - len(subset) - 1)) / fact(args.num_users)) * mc
                banzhaf_values[client] += mc / len(all_subsets)

    longest_client_key = max(results.keys(), key=len)
    test_loss, test_acc, abv_simple, abv_hessian = results[longest_client_key]

    # remove any clients that are not in all metrics
    shared_clients = set(shapley_values.keys()) & set(banzhaf_values.keys()) & set(abv_simple.keys()) & set(abv_hessian.keys())
    sv = [shapley_values[client] for client in shared_clients]
    bv = [banzhaf_values[client] for client in shared_clients]
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
    logger.info(f'Total Run Time: {time.time() - start_time}')