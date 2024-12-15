import copy
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from sampling import iid, noniid, mislabeled, noisy
import logging
import numpy as np
from models import CNNFashion, CNNCifar, ResNet9, ImageNetModel, CNNFashion2
import os
import json
import subprocess
import zipfile


class EarlyStopping:
    def __init__(self, args, patience=3, epoch_threshold=15):
        self.best_acc = -float('inf')
        self.best_loss = float('inf')
        self.no_improvement_count = 0
        self.patience = patience
        self.epoch_threshold = epoch_threshold
        self.acc_threshold = 0.85 if args.dataset in ['fmnist', 'fmnist2'] else 0.80
        self.args = args

    def check(self, epoch, acc, loss):
        if acc > self.best_acc * 1.01 or loss < self.best_loss * 0.99:
            self.best_acc, self.best_loss = acc, loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.args.acc_stopping == 0:
                if self.no_improvement_count > self.patience and epoch > self.epoch_threshold:
                    return True
            else:
                if self.no_improvement_count > self.patience and (epoch > self.epoch_threshold or acc > self.acc_threshold):
                    return True
        return False


class SubsetSplit(Dataset):
    """A Dataset class wrapped around a subset of a PyTorch Dataset."""
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.targets = np.array(self.dataset.targets)[self.idxs].copy()
        self.data = [self.dataset[idx][0].numpy() for idx in self.idxs] 

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image = self.data[item]
        label = self.targets[item]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class AddGaussianNoise(object):
    """Custom transform to add Gaussian noise to a tensor."""
    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean
        self.std = std
        self.device = None
    
    def __call__(self, tensor):
        if self.std > 0:
            noise = torch.randn(tensor.size(), device=self.device) * self.std + self.mean
            return tensor + noise
        return tensor

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

    def set_std(self, std):
        self.std = std

    def to(self, device):
        self.device = device

def get_dataset(args):
    """Returns train, validation, and test datasets along with a user group,
    which is a dict where the keys are the user index and the values are the
    corresponding data for each of those users.
    """
    # define transformations based on dataset
    t_dict = {
        'fmnist': {
            'train': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
            'test': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        },
        'cifar': {
            'train': transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
            'test': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
        },
        'imagenet': {
            'train': transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
            'test': transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        }
    }
    
    # select dataset-specific configurations
    if args.dataset in ['cifar', 'resnet', 'resnet18']:
        dataset_name = 'cifar'
        data_dir = './data/cifar/'
        dataset_class = datasets.CIFAR10
    elif args.dataset in ['fmnist', 'fmnist2']:
        dataset_name = 'fmnist'
        data_dir = './data/fmnist/'
        dataset_class = datasets.FashionMNIST
    elif args.dataset == 'imagenet':
        dataset_name = 'imagenet'
        data_dir = './data/imagenet/'
        dataset_class = datasets.ImageNet

    if dataset_name == 'imagenet':
        if not os.path.exists(data_dir):
            download_imagenet(data_dir)
        train_dataset = dataset_class(data_dir, "train", transform=t_dict[dataset_name]['train'])
        valid_dataset = dataset_class(data_dir, "val", transform=t_dict[dataset_name]['train'])
        test_dataset = dataset_class(data_dir, "test", transform=t_dict[dataset_name]['test'])
    else:
        os.makedirs(data_dir, exist_ok=True)
        full_train_dataset = dataset_class(data_dir, True, download=True, transform=t_dict[dataset_name]['train'])
        train_dataset, valid_dataset = train_val_split(full_train_dataset, 0.1)
        test_dataset = dataset_class(data_dir, False, download=True, transform=t_dict[dataset_name]['test'])

    # handle different settings
    if args.setting == 0:
        user_groups = iid(train_dataset, args.num_users)
        bad_clients = None
    elif args.setting == 1:
        user_groups, bad_clients = noniid(train_dataset, dataset_name, args.num_users, args.badclient_prop, args.num_categories_per_client)
    elif args.setting == 2:
        iid_user_groups = iid(train_dataset, args.num_users)
        user_groups, bad_clients = mislabeled(train_dataset, dataset_name, iid_user_groups, args.badclient_prop, args.badsample_prop)
    elif args.setting == 3:
        iid_user_groups = iid(train_dataset, args.num_users)
        user_groups, bad_clients = noisy(train_dataset, dataset_name, iid_user_groups, args.badclient_prop, args.badsample_prop)
    else:
        raise ValueError("Invalid value for --setting. Please use 0, 1, 2, or 3.")
    return train_dataset, valid_dataset, test_dataset, user_groups, bad_clients


def train_val_split(full_train_dataset, val_prop):
    num_train = len(full_train_dataset)
    split = int(np.floor(val_prop * num_train))
    indices = list(range(num_train))
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    return SubsetSplit(full_train_dataset, train_idx), SubsetSplit(full_train_dataset, valid_idx)


def download_imagenet(data_dir: str):
    os.makedirs(data_dir, exist_ok=True)
    kaggle_api_key = os.getenv('KAGGLE_API_KEY')
    if not kaggle_api_key:
        raise ValueError("KAGGLE_API_KEY is not set as an environment variable")
    kaggle_json = json.loads(kaggle_api_key)
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
    with open(kaggle_json_path, 'w') as f:
        json.dump(kaggle_json, f)
    os.chmod(kaggle_json_path, 0o600)
    subprocess.run(["pip", "install", "--quiet", "kaggle"], check=True)
    subprocess.run(["kaggle", "competitions", "download", "-c", "imagenet-object-localization-challenge", "-p", data_dir], check=True)
    zip_path = os.path.join(data_dir, 'imagenet-object-localization-challenge.zip')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)


def average_weights(w):
    """Returns the average of the weights."""
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))

    del w
    torch.cuda.empty_cache()
    
    return w_avg


def initialize_model(args):
    model_dict = {
        'fmnist': CNNFashion,
        'cifar': CNNCifar,
        'resnet': ResNet9,
        'imagenet': ImageNetModel,
        'fmnist2': CNNFashion2
    }
    if args.dataset in model_dict:
        return model_dict[args.dataset](args=args)
    else:
        exit('Error: unrecognized dataset')


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


def identify_bad_idxs(approx_banzhaf_values: dict, threshold: float = 2) -> list[int]:
    if not approx_banzhaf_values:
        return []
    # add all clients with negative bv or bv less than the mean divided by the threshold
    avg_banzhaf = np.mean(list(approx_banzhaf_values.values()))
    bad_idxs = [key for key, banzhaf in approx_banzhaf_values.items() if banzhaf < avg_banzhaf / threshold or banzhaf < 0]
    return bad_idxs


def measure_accuracy(targets, predictions):
    if targets is None or predictions is None:
        return 0.0
    if len(targets) == 0 and len(predictions) == 0:
        return 1.0
    targets, predictions = set(targets), set(predictions)
    TP = len(predictions & targets)
    FP = len(predictions - targets)
    FN = len(targets - predictions)
    universe = targets | predictions
    TN = len(universe - (targets | predictions))
    return (TP + TN) / (TP + TN + FP + FN)

def visualize_noise():
    import matplotlib.pyplot as plt
    from options import args_parser
    args = args_parser()
    noise_transform = AddGaussianNoise()
    train_dataset, _, _, _, _ = get_dataset(args)

    image, _ = train_dataset[0]

    noise_levels = [0, 0.25, 0.5]
    fig, axes = plt.subplots(1, 3, figsize=(12, 12))

    for col, std in enumerate(noise_levels):
        noise_transform.set_std(std)
        noisy_image_1 = noise_transform(image.unsqueeze(0)).squeeze(0)
        axes[col].imshow(noisy_image_1.permute(1, 2, 0).numpy())
        axes[col].axis("off")

    axes[0].set_title(f"No Noise", fontsize=18)
    axes[1].set_title(f"Noise Drawn From N(0, 0.25)", fontsize=18)
    axes[2].set_title(f"Noise Drawn From N(0, 0.5)", fontsize=18)    

    plt.tight_layout()
    plt.savefig("noise_visualization.png", dpi=300, bbox_inches='tight')