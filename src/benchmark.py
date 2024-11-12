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

    no_improvement_count = 0
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
                b_value = sum((-torch.dot(gradient[key].flatten(), delta_weights[key].flatten()) for key in gradient.keys()))
                approx_banzhaf_values[idx] += b_value.item()

        # update global weights and model
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


if __name__ == '__main__':
    start_time = time.time()
    logger = setup_logger('experiment')
    args = args_parser()
    exp_details(args)

    device = get_device()
    train_dataset, test_dataset, user_groups, _, actual_bad_clients, _ = get_dataset(args)

    # initialize Shapley and Banzhaf values
    shapley_values, banzhaf_values = defaultdict(float), defaultdict(float)
    all_subsets = list(itertools.chain.from_iterable(itertools.combinations(range(args.num_users), r) for r in range(args.num_users + 1)))
    models, results = {}, {}
    print(f"Training Models For {len(all_subsets)} Subsets")
    # train models for each subset
    for subset in all_subsets:
        subset = tuple(sorted(subset))
        models[subset] = train_global_model(args, initialize_model(args).to(device), train_dataset, test_dataset, user_groups, device, subset)[0]
        results[subset] = test_inference(args, models[subset], test_dataset)[0]

    for client in range(args.num_users):
        for r in range(args.num_users):
            for subset in itertools.combinations([c for c in range(args.num_users) if c != client], r):
                subset = tuple(sorted(subset))
                subset_with_client = tuple(sorted(subset + (client,)))
                marginal_contribution = results[subset] - results[subset_with_client]
                shapley_values[client] += ((math.factorial(len(subset)) * math.factorial(args.num_users - len(subset) - 1)) / math.factorial(args.num_users)) * marginal_contribution
                banzhaf_values[client] += marginal_contribution / len(all_subsets)

    global_model = initialize_model(args).to(device)
    global_model.train()
    clients = [c for c in range(args.num_users)]
    global_model, approx_banzhaf_values = train_global_model(args, global_model, train_dataset, test_dataset, user_groups, device, clients=clients, isBanzhaf=True)
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    # log results
    match args.setting:
        case 0:
            setting_str = "IID"
        case 1:
            setting_str = f"{len(actual_bad_clients)} Bad Clients" + f" with {args.num_categories_per_client} Categories Per Bad Client"
        case 2:
            setting_str = f"{len(actual_bad_clients)} Bad Clients" + f" with {100*args.mislabel_proportion}% Mislabeled Samples Per Bad Client"
    logger.info(f'Number Of Clients: {args.num_users}, Client Selection Fraction: {args.frac}, Local Epochs: {args.local_ep}')
    logger.info(f'Dataset: {args.dataset}, Setting: {setting_str}')
    logger.info(f'Test Accuracy Of Global Model: {100*test_acc}%')
    logger.info(f'Shapley Values: {shapley_values}')
    logger.info(f'Banzhaf Values: {banzhaf_values}')
    logger.info(f'Approximate Banzhaf Values: {approx_banzhaf_values}')
    logger.info(f'Pearson Correlation Between Shapley And Banzhaf Values: {pearsonr(list(shapley_values.values()), list(banzhaf_values.values()))}')
    logger.info(f'Spearman Correlation Between Shapley And Banzhaf Values: {spearmanr(list(shapley_values.values()), list(banzhaf_values.values()))}')
    logger.info(f'Pearson Correlation Between Shapley And Approximate Banzhaf Values: {pearsonr(list(shapley_values.values()), list(approx_banzhaf_values.values()))}')
    logger.info(f'Spearman Correlation Between Shapley And Approximate Banzhaf Values: {spearmanr(list(shapley_values.values()), list(approx_banzhaf_values.values()))}')
    logger.info(f'Total Run Time: {time.time() - start_time}')