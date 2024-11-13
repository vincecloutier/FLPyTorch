import copy
import time
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from options import args_parser
from update import LocalUpdate, test_inference, test_gradient
from models import CNNMnist, CNNFashion_Mnist, CNNCifar, ResNet9, MobileNetV2
from utils import get_dataset, average_weights, exp_details, setup_logger, get_device, identify_bad_idxs, measure_accuracy, remove_bad_samples

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

def train_global_model(args, model, train_dataset, test_dataset, user_groups, device, bad_clients=None):
    global_weights = model.state_dict()
    best_test_acc, best_test_loss = 0, float('inf')
    approx_banzhaf_values = defaultdict(float)
    selection_probabilities = np.full(args.num_users, 1 / args.num_users)

    no_improvement_count = 0
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []

        model.train()
        gradient = test_gradient(model, test_dataset)
        
        if bad_clients is not None:
            good_clients = [i for i in range(args.num_users) if i not in bad_clients]
            m = max(int(args.frac * len(good_clients)), 1)
            idxs_users = np.random.choice(good_clients, m, replace=False)
        else:
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)

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

        # update selection probabilities based on the banzhaf values
        if bad_clients is not None:
            total_banzhaf = sum(approx_banzhaf_values.values())
            if total_banzhaf > 0:
                selection_probabilities = np.array([approx_banzhaf_values[i] / total_banzhaf for i in range(args.num_users)])
                selection_probabilities /= selection_probabilities.sum()  # normalize

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
        print(f'Approximate Banzhaf Values: {approx_banzhaf_values}')
        print(f'Local Losses: {local_losses}')
        print(f'Best Test Accuracy: {best_test_acc}, Best Test Loss: {best_test_loss}')
    convergence_round = epoch + 1 if no_improvement_count > 5 else args.epochs
    return model, approx_banzhaf_values, convergence_round


if __name__ == '__main__':
    start_time = time.time()
    logger = setup_logger('experiment')
    args = args_parser()
    exp_details(args)

    device = get_device()
    train_dataset, test_dataset, user_groups, non_iid_clients, actual_bad_clients, actual_bad_samples = get_dataset(args)
    
    # train the global model
    global_model = initialize_model(args)
    global_model.to(device)
    global_model.train()
    global_model, approx_banzhaf_values, convergence_round = train_global_model(args, global_model, train_dataset, test_dataset, user_groups, device)
    test_acc, test_loss = test_inference(global_model, test_dataset)

    # predict bad clients and measure accuracy
    predicted_bad_clients = identify_bad_idxs(approx_banzhaf_values)
    bad_client_accuracy = measure_accuracy(actual_bad_clients, predicted_bad_clients)

    # retrain the model w/o bad clients 
    global_model = initialize_model(args)
    global_model.to(device)
    global_model.train()
    retrained_model, _, second_convergence_round = train_global_model(args, global_model, train_dataset, test_dataset, user_groups, device, predicted_bad_clients)
    retrain_test_acc, retrain_test_loss = test_inference(retrained_model, test_dataset)

    # log results
    match args.setting:
        case 0:
            setting_str = "IID"
        case 1:
            setting_str = f"{len(actual_bad_clients)} Bad Clients" + f" with {args.num_categories_per_client} Categories Per Bad Client"
        case 2:
            setting_str = f"{len(actual_bad_clients)} Bad Clients" + f" with {100*args.mislabel_proportion}% Mislabeled Samples Per Bad Client"
    logger.info(f'Number Of Clients: {args.num_users}, Client Selection Fraction: {args.frac}, Local Epochs: {args.local_ep}')
    logger.info(f'Batch Size: {args.local_bs}, Learning Rate: {args.lr}, Momentum: {args.momentum}')
    logger.info(f'Convergence Round: {convergence_round}, Retraining Convergence Round: {second_convergence_round}, Number Of Rounds: {args.epochs}')
    logger.info(f'Dataset: {args.dataset}, Setting: {setting_str}')
    logger.info(f'Test Accuracy Before Retraining: {100*test_acc}%')
    logger.info(f'Test Accuracy After Retraining: {100*retrain_test_acc}%')
    logger.info(f'Bad Client Accuracy: {bad_client_accuracy}')
    logger.info(f'Total Run Time: {time.time()-start_time}')