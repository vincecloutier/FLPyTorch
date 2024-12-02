import copy
import time
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from options import args_parser
from update import LocalUpdate, test_inference, gradient
from utils import get_dataset, average_weights, setup_logger, get_device, identify_bad_idxs, measure_accuracy, initialize_model
from valuation.banzhaf import compute_abv, compute_G_t, compute_G_minus_i_t
import warnings
import multiprocessing
from functools import partial
warnings.filterwarnings("ignore", category=UserWarning)
import os

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def train_client(idx, args, global_weights, train_dataset, user_groups, epoch, device):
    # torch.cuda.set_device(device)

    model = initialize_model(args)
    model.load_state_dict(global_weights)
    model.to(device)
    model.train()

    local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
    w, _ = local_model.update_weights(model=model, global_round=epoch)
    delta = {key: (global_weights[key] - w[key]).to(device) for key in global_weights.keys()}

    del model, local_model
    torch.cuda.empty_cache()

    return idx, w, delta


def train_global_model(args, model, train_dataset, valid_dataset, test_dataset, user_groups, device, bad_clients=None):
    global_weights = model.state_dict()
    best_test_acc, best_test_loss = 0, float('inf')
    approx_banzhaf_values = defaultdict(float)
    selection_probabilities = np.full(args.num_users, 1 / args.num_users)
    delta_t = defaultdict(dict)
    delta_g = defaultdict(lambda: {key: torch.zeros_like(global_weights[key]) for key in global_weights.keys()})

    no_improvement_count = 0
    for epoch in tqdm(range(args.epochs)):
        local_weights = []

        model.train()
        grad = gradient(args, model, valid_dataset)
        
        if bad_clients is not None:
            good_clients = [i for i in range(args.num_users) if i not in bad_clients]
            m = max(int(args.frac * len(good_clients)), 1)
            idxs_users = np.random.choice(good_clients, m, replace=False)
        else:
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        train_client_partial = partial(train_client, args=args, global_weights=copy.deepcopy(global_weights), train_dataset=train_dataset, user_groups=user_groups, epoch=epoch, device=device)

        with multiprocessing.Pool(processes=args.processes) as pool:
            results = pool.map(train_client_partial, idxs_users)
        pool.close()
        pool.join()
        for idx, w, delta in results:
            local_weights.append(copy.deepcopy(w))
            delta_t[epoch][idx] = delta

        global_weights = average_weights(local_weights)
        model.load_state_dict(global_weights)

        # compute banzhaf values
        if args.hessian == 1:
            G_t = compute_G_t(delta_t[epoch], global_weights.keys())
            for idx in idxs_users:
                G_t_minus_i = compute_G_minus_i_t(delta_t[epoch], global_weights.keys(), idx)
                if epoch > 0:
                    for key in global_weights.keys():
                        delta_g[idx][key] += G_t_minus_i[key] - G_t[key]
                approx_banzhaf_values[idx] += compute_abv(args, model, train_dataset, grad, delta_t[epoch][idx], delta_g[idx], is_hessian=True)
        else:
            for idx in idxs_users:
                approx_banzhaf_values[idx] += compute_abv(args, model, train_dataset, grad, delta_t[epoch][idx], delta_g[idx], is_hessian=False)

        # update selection probabilities based on the banzhaf values
        if bad_clients is not None:
            total_banzhaf = sum(approx_banzhaf_values.values())
            if total_banzhaf > 0:
                selection_probabilities = np.array([approx_banzhaf_values[i] / total_banzhaf for i in range(args.num_users)])
                selection_probabilities /= selection_probabilities.sum()  # normalize

        test_acc, test_loss = test_inference(model, test_dataset)
        if test_acc > best_test_acc * 1.01 or test_loss < best_test_loss * 0.99:
            best_test_acc, best_test_loss = test_acc, test_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if (no_improvement_count > 3 and epoch > 20) or test_acc > 0.80:
                print(f'Convergence Reached At Round {epoch + 1}')
                break

        print(f'Epoch {epoch+1}/{args.epochs} - Test Accuracy: {test_acc}, Test Loss: {test_loss}')
        # print(torch.cuda.memory_summary(device=device))
            
    return model, approx_banzhaf_values


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    # if torch.cuda.is_available():
    #     torch.backends.cudnn.benchmark = True
    #     torch.backends.cudnn.enabled = True

    start_time = time.time()
    logger = setup_logger('experiment')
    args = args_parser()
    print(args)

    device = get_device()
    train_dataset, valid_dataset, test_dataset, user_groups, actual_bad_clients = get_dataset(args)
    
    # torch.cuda.set_per_process_memory_fraction(0.25, device)

    # train the global model
    global_model = initialize_model(args)
    global_model.to(device)
    global_model.train()
    global_model, approx_banzhaf_values = train_global_model(args, global_model, train_dataset, valid_dataset, test_dataset, user_groups, device)
    test_acc, test_loss = test_inference(global_model, test_dataset)

    # predict bad clients and measure accuracy
    predicted_bad_clients = identify_bad_idxs(approx_banzhaf_values)
    bad_client_accuracy = measure_accuracy(actual_bad_clients, predicted_bad_clients)

    # retrain the model w/o bad clients 
    if args.retrain:
        global_model = initialize_model(args)
        global_model.to(device)
        global_model.train()
        retrained_model, _, = train_global_model(args, global_model, train_dataset, valid_dataset, test_dataset, user_groups, device, predicted_bad_clients)
        retrain_test_acc, retrain_test_loss = test_inference(retrained_model, test_dataset)

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
    logger.info(f'Dataset: {args.dataset}, Setting: {setting_str}, Number Of Rounds: {args.epochs}, Hessian: {args.hessian}')
    logger.info(f'Test Accuracy Before Retraining: {100*test_acc}%')
    if args.retrain:
        logger.info(f'Test Accuracy After Retraining: {100*retrain_test_acc}%')
    logger.info(f'Banzhaf Values: {approx_banzhaf_values}')
    logger.info(f'Actual Bad Clients: {actual_bad_clients}')
    logger.info(f'Predicted Bad Clients: {predicted_bad_clients}')
    logger.info(f'Bad Client Accuracy: {bad_client_accuracy}')
    logger.info(f'Total Run Time: {time.time()-start_time}')