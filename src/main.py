import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch

from options import args_parser
from update import LocalUpdate, test_inference
from models import CNNMnist, CNNFashion_Mnist, CNNCifar, ResNet9
from utils import get_dataset, average_weights, exp_details, setup_logger


if __name__ == '__main__':
    start_time = time.time()
    logger = setup_logger('experiment')
    args = args_parser()
    exp_details(args)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'mps'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    if args.dataset == 'mnist':
        global_model = CNNMnist(args=args)
    elif args.dataset == 'fmnist':
        global_model = CNNFashion_Mnist(args=args)
    elif args.dataset == 'cifar':
        global_model = CNNCifar(args=args)
    elif args.dataset == 'resnet':
        global_model = ResNet9(args=args)
    else:
        exit('Error: unrecognized dataset') 

    # set the model to train and send it to device.
    global_model.to(device)
    global_model.train()

    # copy weights
    global_weights = global_model.state_dict()

    # training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        # compute banzhaf values
        # banzhaf_values = compute_banzhaf(global_weights, train_dataset, args.num_users)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    match args.setting:
        case 0:
            setting_str = "IID"
        case 1:
            setting_str = "non-iid" + f" with {args.num_categories_per_client} categories per client" + f" and {args.badclient_prop} bad clients"
        case 2:
            setting_str = "mislabeled" + f" with {args.mislabel_proportion} mislabeled samples per client" + f" and {args.badclient_prop} bad clients"
    logger.info(f'Results after {args.epochs} global rounds of training model {args.dataset} in {setting_str}:')
    logger.info(f'Avg Train Accuracy: {100*train_accuracy[-1]}%')
    logger.info(f'Test Accuracy: {100*test_acc}%')
    logger.info(f'Total Run Time: {time.time()-start_time}')
