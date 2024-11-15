import copy
import torch
from torchvision import datasets, transforms
from sampling import iid, noniid, mislabeled
import logging
import numpy as np

def get_dataset(args):
    """ Returns train, validation, and test datasets along with a user group,
    which is a dict where the keys are the user index and the values are the
    corresponding data for each of those users.
    """
    if args.dataset == 'cifar' or args.dataset == 'resnet' or args.dataset == 'mobilenet':
        data_dir = './data/cifar/'
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        # load the full training dataset
        full_train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)

        # allocate 10% of the training set as validation set
        num_train = len(full_train_dataset)
        split = int(np.floor(0.1 * num_train))
        indices = list(range(num_train))
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]

        # create train and validation datasets
        train_dataset = torch.utils.data.Subset(full_train_dataset, train_idx)
        valid_dataset = torch.utils.data.Subset(full_train_dataset, valid_idx)

        # load the test dataset
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
    elif args.dataset == 'mnist' or args.dataset == 'fmnist':
        data_dir = './data/mnist/' if args.dataset == 'mnist' else './data/fmnist/'
        apply_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        # load the full training dataset
        full_train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)

        # allocate 10% of the training set as validation set
        num_train = len(full_train_dataset)
        split = int(np.floor(0.1 * num_train))
        indices = list(range(num_train))
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]

        # create train and validation datasets
        train_dataset = torch.utils.data.Subset(full_train_dataset, train_idx)
        valid_dataset = torch.utils.data.Subset(full_train_dataset, valid_idx)

        # load the test dataset
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)
     
    match args.setting:
        case 0:
            user_groups = iid(train_dataset, args.num_users)
            return train_dataset, valid_dataset, test_dataset, user_groups, None
        case 1:
            user_groups, bad_clients = noniid(train_dataset, args.dataset, args.num_users, args.badclient_prop, args.num_categories_per_client)
            return train_dataset, valid_dataset, test_dataset, user_groups, bad_clients
        case 2:
            iid_user_groups = iid(train_dataset, args.num_users)
            user_groups, bad_clients = mislabeled(train_dataset, args.dataset, iid_user_groups, args.badclient_prop, args.mislabel_proportion)
            return train_dataset, valid_dataset, test_dataset, user_groups, bad_clients
        case _:
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
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')

def identify_bad_idxs(approx_banzhaf_values: dict, threshold: float = 1.5) -> list[int]:
    if not approx_banzhaf_values:
        return []
    banzhaf_tensor = torch.tensor(list(approx_banzhaf_values.values()))
    median_banzhaf = torch.median(banzhaf_tensor)    
    bad_idxs = [key for key, banzhaf in approx_banzhaf_values.items() if banzhaf < median_banzhaf / threshold]
    return bad_idxs

def measure_accuracy(targets, predictions):
    if targets is None or predictions is None:
        return 0.0
    if len(targets) == 0 or len(predictions) == 0:
        return 0.0
    targets, predictions = set(targets), set(predictions)
    TP = len(predictions & targets)
    FP = len(predictions - targets)
    FN = len(targets - predictions)
    TN = len(targets) - (TP + FP + FN)
    return (TP + TN) / len(targets)