import copy
import torch
from torchvision import datasets, transforms
from sampling import iid, noniid
import logging

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    if args.dataset == 'cifar' or args.dataset == 'resnet':
        data_dir = './data/cifar/'
        apply_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)
    elif args.dataset == 'mnist' or args.dataset == 'fmnist':
        data_dir = './data/mnist/' if args.dataset == 'mnist' else './data/fmnist/'
        apply_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)
 
    # sample training data amongst users
    if args.iid:
        user_groups = iid(train_dataset, args.num_users)
    else:
        user_groups = noniid(train_dataset, args.dataset, args.num_users, 0.4, 4)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """Returns the average of the weights."""
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.dataset}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


def setup_logger(strategy_name: str) -> logging.Logger:
    """Set up a logger for the given strategy."""
    logger = logging.getLogger(strategy_name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f'{strategy_name}_metrics.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(fh)
    return logger