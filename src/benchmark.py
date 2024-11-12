import itertools
import math
from collections import defaultdict
import copy
import time
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from options import args_parser
from update import LocalUpdate, test_inference, test_gradient
from models import CNNMnist, CNNFashion_Mnist, CNNCifar, ResNet9
from utils import get_dataset, average_weights, exp_details, setup_logger, get_device, identify_bad_idxs, measure_accuracy, remove_bad_samples
from scipy.stats import pearsonr, spearmanr

def initialize_model(args):
    model_dict = {
        'mnist': CNNMnist,
        'fmnist': CNNFashion_Mnist,
        'cifar': CNNCifar,
        'resnet': ResNet9
    }
    if args.dataset in model_dict:
        return model_dict[args.dataset](args=args)
    else:
        exit('Error: unrecognized dataset')

def train_global_model(args, model, train_dataset, test_dataset, user_groups, device, clients = [int]):
    global_weights = model.state_dict()
    train_loss, train_accuracy = [], []
    best_test_acc, best_test_loss = 0, float('inf')
    approx_banzhaf_values = defaultdict(float)

    no_improvement_count = 0
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} w/ {len(clients)} Clients |\n')

        model.train()
        gradient = test_gradient(model, test_dataset)
        
        m = max(int(args.frac * len(clients)), 1)
        idxs_users = np.random.choice(clients, m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss = local_model.update_weights(model=copy.deepcopy(model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

            # compute banzhaf value estimate
            delta_weights = {key: (w[key] - global_weights[key]).to(device) for key in w.keys()}
            b_value = sum((-torch.dot(gradient[key].flatten(), delta_weights[key].flatten()) for key in gradient.keys()))
            approx_banzhaf_values[idx] += b_value.item()

        # update global weights and model
        global_weights = average_weights(local_weights)
        model.load_state_dict(global_weights)
        
        # calculate average loss
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # calculate average training accuracy
        acc = evaluate_model(args, model, train_dataset, user_groups)
        train_accuracy.append(acc)
        
        if (epoch + 1) % 2 == 0:
            print(f'\nAvg Training Stats after {epoch + 1} global rounds:')
            print(f'Training Loss: {np.mean(train_loss)}')
            print(f'Train Accuracy: {100 * acc:.2f}% \n')

        test_acc, test_loss = test_inference(args, model, test_dataset)
        if test_acc > best_test_acc * 1.01 or test_loss < best_test_loss * 0.99:
            best_test_acc = test_acc
            best_test_loss = test_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count > 5:
                print(f'Convergence Reached At Round {epoch + 1}')
                break
            
    return model, approx_banzhaf_values    

def evaluate_model(args, model, train_dataset, user_groups):
    model.eval()
    list_acc = []
    for c in range(args.num_users):
        local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[c])
        acc, _ = local_model.inference(model=model)
        list_acc.append(acc)
    return sum(list_acc) / len(list_acc)

if __name__ == '__main__':
    start_time = time.time()
    logger = setup_logger('experiment')
    args = args_parser()
    exp_details(args)

    device = get_device()
    train_dataset, test_dataset, user_groups, non_iid_clients, actual_bad_clients, actual_bad_samples = get_dataset(args)
    
    # calculate shapley/banzhaf values
    shapley_values = defaultdict(float)
    banzhaf_values = defaultdict(float)
    all_subsets = list(itertools.chain.from_iterable(itertools.combinations(range(args.num_users), r) for r in range(args.num_users + 1)))
    all_subsets = {tuple(sorted([c for c in subset])): subset for subset in all_subsets}
    models = {a: train_global_model(args, global_model, train_dataset, test_dataset, user_groups, device, subset)[0] for subset in all_subsets}
    results = {a: test_inference(args, models[a], test_dataset)[0] for a in all_subsets}
    
    for client in range(args.num_users):
        for r in range(args.num_users):  
            for subset in itertools.combinations([c for c in range(args.num_users) if c != client], r):
                subset = tuple(sorted([c for c in subset]))
                subset_with_client = tuple(sorted(subset + (client,)))
                marginal_contribution = results[subset] - results[subset_with_client]
                shapley_values[client] += ((math.factorial(len(subset)) * math.factorial(args.num_users - len(subset) - 1)) / math.factorial(args.num_users)) * marginal_contribution
                banzhaf_values[client] += marginal_contribution / len(all_subsets)

    # estimate banzhaf values
    global_model = initialize_model(args)
    global_model.to(device)
    global_model.train()
    global_model, approx_banzhaf_values, convergence_round = train_global_model(args, global_model, train_dataset, test_dataset, user_groups, device)
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
    logger.info(f'Convergence Round: {convergence_round}, Number Of Rounds: {args.epochs}')
    logger.info(f'Dataset: {args.dataset}, Setting: {setting_str}')
    logger.info(f'Test Accuracy Of Global Model: {100*test_acc}%')
    logger.info(f'Shapley Values: {shapley_values}')
    logger.info(f'Banzhaf Values: {banzhaf_values}')
    logger.info(f'Approximate Banzhaf Values: {approx_banzhaf_values}')
    logger.info(f'Pearson Correlation Between Shapley And Banzhaf Values: {pearsonr(list(shapley_values.values()), list(banzhaf_values.values()))}')
    logger.info(f'Spearman Correlation Between Shapley And Banzhaf Values: {spearmanr(list(shapley_values.values()), list(banzhaf_values.values()))}')
    logger.info(f'Pearson Correlation Between Shapley And Approximate Banzhaf Values: {pearsonr(list(shapley_values.values()), list(approx_banzhaf_values.values()))}')
    logger.info(f'Spearman Correlation Between Shapley And Approximate Banzhaf Values: {spearmanr(list(shapley_values.values()), list(approx_banzhaf_values.values()))}')
    logger.info(f'Pearson Correlation Between Banzhaf And Approximate Banzhaf Values: {pearsonr(list(banzhaf_values.values()), list(approx_banzhaf_values.values()))}')
    logger.info(f'Spearman Correlation Between Banzhaf And Approximate Banzhaf Values: {spearmanr(list(banzhaf_values.values()), list(approx_banzhaf_values.values()))}')
    logger.info(f'Total Run Time: {time.time()-start_time}')