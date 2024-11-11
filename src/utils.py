import copy
import torch
from torchvision import datasets, transforms
from sampling import iid, noniid, mislabeled
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
     
    match args.setting:
        case 0:
            user_groups = iid(train_dataset, args.num_users)
            return train_dataset, test_dataset, user_groups, None, None, None
        case 1:
            user_groups, non_iid_clients = noniid(train_dataset, args.dataset, args.num_users, args.badclient_prop, args.num_categories_per_client)
            return train_dataset, test_dataset, user_groups, non_iid_clients, None, None
        case 2:
            iid_user_groups = iid(train_dataset, args.num_users)
            user_groups, bad_clients, bad_samples = mislabeled(train_dataset, args.dataset, iid_user_groups, args.badclient_prop, args.mislabel_proportion)
            return train_dataset, test_dataset, user_groups, None, bad_clients, bad_samples
        case _  :
            raise ValueError("Invalid value for --iid. Please use 0 or 1.")


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
    if args.setting:
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

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'mps'

def identify_bad_idxs(approx_banzhaf_values: dict, threshold: float = 1.5) -> list[int]:
    if not approx_banzhaf_values:
        return []
    banzhaf_tensor = torch.tensor(list(approx_banzhaf_values.values()))
    median_banzhaf = torch.median(banzhaf_tensor)    
    bad_idxs = [key for key, banzhaf in approx_banzhaf_values.items() if banzhaf < median_banzhaf / threshold]
    return bad_idxs

def measure_accuracy(targets, predictions):
    targets, predictions = set(targets), set(predictions)
    TP = len(predictions & targets)
    FP = len(predictions - targets)
    FN = len(targets - predictions)
    TN = len(targets) - (TP + FP + FN)
    return (TP + TN) / len(targets) if len(targets) > 0 else 0.0

def remove_bad_samples(user_groups, bad_samples):
    updated_user_groups = {}
    for client_idx, data_idxs in user_groups.items():
        updated_data_idxs = [idx for idx in data_idxs if idx not in bad_samples]
        updated_user_groups[client_idx] = updated_data_idxs
    return updated_user_groups