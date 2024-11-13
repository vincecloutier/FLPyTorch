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
import multiprocessing
from scipy.stats import pearsonr, spearmanr
from functools import partial


def initialize_model(args):
    model_dict = {
        'mnist': CNNMnist,
        'fmnist': CNNFashion_Mnist,
        'cifar': CNNCifar,
        'resnet': ResNet9,
        'mobilenet': MobileNetV2
    }
    if args.dataset in model_dict:
        return model_dict[args.dataset](args=args)
    else:
        exit('Error: unrecognized dataset')


def train_global_model(args, model, train_dataset, test_dataset, user_groups, device, clients=None, isBanzhaf=False):
    if clients is None or len(clients) == 0:
        return model, defaultdict(float)
    global_weights = model.state_dict()
    best_test_acc, best_test_loss = 0, float('inf')
    approx_banzhaf_values = defaultdict(float)

    for epoch in tqdm(range(args.epochs), desc=f"Global Training For Subset {clients}"):
        local_weights, local_losses = [], []

        model.train()
        if isBanzhaf:
            gradient = test_gradient(model, test_dataset)

        m = max(int(args.frac * len(clients)), 1)
        idxs_users = np.random.choice(clients, m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss = local_model.update_weights(model=copy.deepcopy(model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(loss)

            # compute banzhaf value estimate
            if isBanzhaf:
                delta_weights = {key: (w[key] - global_weights[key]).to(device) for key in w.keys()}
                # b_value = sum((torch.dot(gradient[key].flatten(), delta_weights[key].flatten()) for key in gradient.keys()))
                b_value = sum((gradient[key] * delta_weights[key]).sum() for key in gradient.keys())
                approx_banzhaf_values[idx] -= b_value.item() / args.num_users


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

    # clients_str = "_".join(str(client) for client in clients)
    # torch.save(model.state_dict(), f"{clients_str}_epoch_{epoch + 1}_{args.dataset}_{args.setting}.pth")
    return model, approx_banzhaf_values


def train_subset(subset, args, train_dataset, test_dataset, user_groups):
    device = get_device()
    global_model = initialize_model(args)
    global_model.to(device)
    global_model.train()

    subset_key = tuple(sorted(subset))
    print(f"Training Model For Subset {subset_key}")
    model, _ = train_global_model(args, global_model, train_dataset, test_dataset, user_groups, device, subset)
    accuracy, loss = test_inference(model, test_dataset)
    torch.cuda.empty_cache()
    return (subset_key, loss)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    start_time = time.time()
    args = args_parser()
    logger = setup_logger(f'benchmark_{args.dataset}_{args.setting}_{args.num_users}_{args.local_bs}')
    exp_details(args)

    device = get_device()
    train_dataset, test_dataset, user_groups, _, _, _ = get_dataset(args)

    shapley_values, banzhaf_values = defaultdict(float), defaultdict(float)
    all_subsets = list(itertools.chain.from_iterable(itertools.combinations(range(args.num_users), r) for r in range(args.num_users + 1)))


    pool = multiprocessing.Pool(processes=18)
    train_subset_partial = partial(train_subset, args=args, train_dataset=train_dataset, test_dataset=test_dataset, user_groups=user_groups)
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
    global_model, approx_banzhaf_values = train_global_model(args, global_model, train_dataset, test_dataset, user_groups, device, clients=clients, isBanzhaf=True)
    test_acc, test_loss = test_inference(global_model, test_dataset)
    
    # remove any clients that are not in approx_banzhaf_values and are not in shapley_values and banzhaf_values 
    shared_clients = set(shapley_values.keys()) & set(banzhaf_values.keys()) & set(approx_banzhaf_values.keys())
    shapley_values = [shapley_values[client] for client in shared_clients]
    banzhaf_values = [banzhaf_values[client] for client in shared_clients]
    approx_banzhaf_values = [approx_banzhaf_values[client] for client in shared_clients]
    print(shapley_values)
    print(banzhaf_values)
    print(approx_banzhaf_values)

    # log results
    logger.info(f'Number Of Clients: {args.num_users}, Client Selection Fraction: {args.frac}, Local Epochs: {args.local_ep}, Batch Size: {args.local_bs}')
    logger.info(f'Dataset: {args.dataset}, Setting: IID, Number Of Rounds: {args.epochs}')
    logger.info(f'Test Accuracy Of Global Model: {100*test_acc}%')
    logger.info(f'Shapley Values: {shapley_values}')
    logger.info(f'Banzhaf Values: {banzhaf_values}')
    logger.info(f'Approximate Banzhaf Values: {approx_banzhaf_values}')
    logger.info(f'Pearson Correlation Between Shapley And Banzhaf Values: {pearsonr(shapley_values, banzhaf_values)}')
    logger.info(f'Spearman Correlation Between Shapley And Banzhaf Values: {spearmanr(shapley_values, banzhaf_values)}')
    logger.info(f'Pearson Correlation Between Shapley And Approximate Banzhaf Values: {pearsonr(shapley_values, approx_banzhaf_values)}')
    logger.info(f'Spearman Correlation Between Shapley And Approximate Banzhaf Values: {spearmanr(shapley_values, approx_banzhaf_values)}')