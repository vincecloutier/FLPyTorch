import copy
import time
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from options import args_parser
from update import LocalUpdate, test_inference, test_gradient
from models import CNNMnist, CNNFashion_Mnist, CNNCifar, ResNet9
from utils import get_dataset, average_weights, exp_details, setup_logger, get_device, identify_bad_clients, measure_bad_client_accuracy

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

def train_global_model(args, model, train_dataset, test_dataset, user_groups, device, bad_clients=None):
    global_weights = model.state_dict()
    train_loss, train_accuracy = [], []
    approx_banzhaf_values = defaultdict(float)
    
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        model.train()
        gradient = test_gradient(model, test_dataset)
        for key in gradient:
            gradient[key] = gradient[key].detach().to(device)
        
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
            
            # compute Banzhaf value estimate
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
    train_dataset, test_dataset, user_groups = get_dataset(args)
    
    # train the global model
    global_model = initialize_model(args)
    global_model.to(device)
    global_model.train()
    global_model, approx_banzhaf_values = train_global_model(args, global_model, train_dataset, test_dataset, user_groups, device)
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    # identify bad clients and retrain
    bad_clients = identify_bad_clients(approx_banzhaf_values)
    bad_client_accuracy = measure_bad_client_accuracy(args.num_users, bad_clients, args.badclient_prop) 

    # retrain global model without bad clients
    global_model = initialize_model(args)
    global_model.to(device)
    global_model.train()
    retrained_model, _ = train_global_model(args, global_model, train_dataset, test_dataset, user_groups, device, bad_clients)
    retrain_test_acc, retrain_test_loss = test_inference(args, retrained_model, test_dataset)

    # log results
    match args.setting:
        case 0:
            setting_str = "IID"
        case 1:
            setting_str = "non-iid" + f" with {args.num_categories_per_client} categories per client" + f" and {args.badclient_prop} bad clients proportion"
        case 2:
            setting_str = "mislabeled" + f" with {args.mislabel_proportion} mislabeled sample proportion per client" + f" and {args.badclient_prop} bad clients proportion"
    logger.info(f'Results after {args.epochs} global rounds of training model {args.dataset} in {setting_str}:')
    logger.info(f'Test Accuracy before retraining: {100*test_acc}% and after retraining: {100*retrain_test_acc}%')
    logger.info(f'Test loss before retraining: {test_loss} and after retraining: {retrain_test_loss}')
    logger.info(f'Bad client accuracy: {bad_client_accuracy}')
    logger.info(f'Total Run Time: {time.time()-start_time}')
