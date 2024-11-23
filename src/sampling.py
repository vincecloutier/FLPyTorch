import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from sampling import iid, noniid, mislabeled, noisy
import numpy as np
import os
import json
import subprocess
import torch
from torch.utils.data import DataLoader


class SubsetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class."""
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.data = dataset.data
        self.idxs = [int(i) for i in idxs]
        self.targets = np.array(self.dataset.targets)[self.idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if isinstance(label, torch.Tensor):
            label = label.clone().detach().long()
        else:
            label = torch.as_tensor(label, dtype=torch.long, device = image.device)
        return image, label


def get_dataset(args):
    """ Returns train, validation, and test datasets along with a user group,
    which is a dict where the keys are the user index and the values are the
    corresponding data for each of those users.
    """
    if args.dataset == 'cifar' or args.dataset == 'resnet' or args.dataset == 'mobilenet':
        data_dir = './data/cifar/'
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        # load the full training dataset
        full_train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)

        # allocate 10% of the training set as validation set
        train_dataset, valid_dataset = train_val_split(full_train_dataset, 0.1)

        # load the test dataset
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)

    elif args.dataset == 'fmnist':
        data_dir = './data/fmnist/'
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # load the full training dataset
        full_train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=train_transform)

        # create train and validation datasets
        train_dataset, valid_dataset = train_val_split(full_train_dataset, 0.1)

        # load the test dataset
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=test_transform)
        
    else:
        data_dir = './data/imagenet/'
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # download the dataset if it doesn't exist
        download_imagenet(data_dir) 

        # load the full training dataset
        full_train_dataset = datasets.ImageNet(data_dir, train=True, transform=train_transform)

        # allocate 10% of the training set as validation set
        train_dataset, valid_dataset = train_val_split(full_train_dataset, 0.1)

        # load the test dataset
        test_dataset = datasets.ImageNet(data_dir, train=False, transform=test_transform)

    # handle different settings
    if args.setting == 0:
        user_groups = iid(train_dataset, args.num_users)
        return train_dataset, valid_dataset, test_dataset, user_groups, None
    elif args.setting == 1:
        user_groups, bad_clients = noniid(train_dataset, args.dataset, args.num_users, args.badclient_prop, args.num_categories_per_client)
        return train_dataset, valid_dataset, test_dataset, user_groups, bad_clients
    elif args.setting == 2:
        iid_user_groups = iid(train_dataset, args.num_users)
        user_groups, bad_clients = mislabeled(train_dataset, args.dataset, iid_user_groups, args.badclient_prop, args.badsample_prop)
        return train_dataset, valid_dataset, test_dataset, user_groups, bad_clients
    elif args.setting == 3:
        iid_user_groups = iid(train_dataset, args.num_users)
        user_groups, bad_clients = noisy(train_dataset, args.dataset, iid_user_groups, args.badclient_prop, args.badsample_prop)
        return train_dataset, valid_dataset, test_dataset, user_groups, bad_clients 
    else:
        raise ValueError("Invalid value for --setting. Please use 0, 1, 2, or 3.")


def download_imagenet(data_dir: str):
    # create the data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # fetch the kaggle key from the environment variable
    kaggle_api_key = os.getenv('KAGGLE_API_KEY')
    if not kaggle_api_key:
        raise ValueError("KAGGLE_API_KEY is not set as an environment variable")

    # parse the key
    kaggle_json = json.loads(kaggle_api_key)

    # create the ~/.kaggle directory if it doesn't exist
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)

    # write the kaggle.json file
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
    with open(kaggle_json_path, 'w') as f:
        json.dump(kaggle_json, f)

    # set correct permissions
    os.chmod(kaggle_json_path, 0o600)

    # ensure kaggle cli is installed
    subprocess.run(["pip", "install", "--quiet", "kaggle"], check=True)

    # download data in parallel
    subprocess.run(["kaggle", "competitions", "download", "-c", "imagenet-object-localization-challenge", "-p", data_dir, "--force"], check=True)

def train_val_split(full_train_dataset, val_prop):
    num_train = len(full_train_dataset)
    split = int(np.floor(val_prop * num_train))
    indices = list(range(num_train))
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    return SubsetSplit(full_train_dataset, train_idx), SubsetSplit(full_train_dataset, valid_idx)


def create_train_loaders(args, dataset, user_groups):
    return {idx: DataLoader(SubsetSplit(dataset, user_groups[idx]), batch_size=args.local_bs, shuffle=True, num_workers=args.num_workers) for idx in user_groups}



def iid(dataset, num_users):
    """Sample iid client data."""
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def noniid(dataset, dataset_name, num_users, badclient_prop, num_cat):
    """Sample non-IID client data with specified number of categories per client and percentage of non-IID clients."""
    if dataset_name == 'fmnist':
        shard_size = 100
    else:
        shard_size = 250

    idxs = np.arange(len(dataset.targets))
    num_classes = len(np.unique(dataset.targets))

    # create shards per category
    shards_per_category = {}
    shard_id_to_indices = {}
    shard_id = 0
    for c in range(num_classes):
        idxs_c = idxs[dataset.targets == c]
        np.random.shuffle(idxs_c)
        num_shards_c = len(idxs_c) // shard_size
        shards_c = []
        for i in range(num_shards_c):
            shard_indices = idxs_c[i * shard_size: (i + 1) * shard_size]
            shard_id_to_indices[shard_id] = shard_indices
            shards_c.append(shard_id)
            shard_id += 1
        shards_per_category[c] = shards_c

    # collect all shard ids
    all_shard_ids = list(range(shard_id))
    np.random.shuffle(all_shard_ids)

    # initialize user data dictionary
    dict_users = {i: np.array([], dtype=int) for i in range(num_users)}

    num_non_iid_clients = int(badclient_prop * num_users)
    num_iid_clients = num_users - num_non_iid_clients
    shards_per_client = shard_id // num_users
    iid_clients = np.random.choice(range(num_users), num_iid_clients, replace=False)
    non_iid_clients = [i for i in range(num_users) if i not in iid_clients]

    # assign shards to iid clients
    assigned_shard_ids = set()
    for i in iid_clients:
        client_shard_ids = all_shard_ids[:shards_per_client]
        all_shard_ids = all_shard_ids[shards_per_client:]
        dict_users[i] = np.concatenate([shard_id_to_indices[sid] for sid in client_shard_ids])
        assigned_shard_ids.update(client_shard_ids)

    # assign shards to non-IID clients
    for i in non_iid_clients:
        # select k random categories
        categories = np.random.choice(num_classes, num_cat, replace=False)
        available_shards = []
        for c in categories:
            available_shards.extend([sid for sid in shards_per_category[c] if sid not in assigned_shard_ids])

        np.random.shuffle(available_shards)
        client_shard_ids = available_shards[:shards_per_client]
        dict_users[i] = np.concatenate([shard_id_to_indices[sid] for sid in client_shard_ids])
        assigned_shard_ids.update(client_shard_ids)

    return dict_users, non_iid_clients


def mislabeled(dataset, dataset_name, dict_users, badclient_prop, mislabel_prop):
    """Randomly select a proportion of clients and mislabel a proportion of their samples."""
    labels = dataset.targets.copy()
    clients_to_mislabel = np.random.choice(range(len(dict_users)), int(badclient_prop * len(dict_users)), replace=False)
    for client_id in clients_to_mislabel:
        client_indices = np.array(list(dict_users[client_id]), dtype=int)
        indices_to_mislabel = np.random.choice(client_indices, int(mislabel_prop * len(client_indices)), replace=False)
        for idx in indices_to_mislabel:
            correct_label = labels[idx]
            incorrect_labels = list(range(10))
            incorrect_labels.remove(correct_label)
            new_label = np.random.choice(incorrect_labels)
            labels[idx] = new_label
    dataset.targets = labels
    return dict_users, clients_to_mislabel


def noisy(dataset, dataset_name, dict_users, badclient_prop, noisy_proportion):
    """Randomly select a proportion of clients and add noise to a proportion of their samples."""
    labels = dataset.targets.copy()
    clients_to_noisy = np.random.choice(range(len(dict_users)), int(badclient_prop * len(dict_users)), replace=False)
    for client_id in clients_to_noisy:
        client_indices = np.array(list(dict_users[client_id]), dtype=int)
        indices_of_base_images = np.where(labels[client_indices] != 2)[0]
        indices_of_base_images = np.random.choice(indices_of_base_images, int(noisy_proportion * len(indices_of_base_images)), replace=False)
        indices_of_target_class = np.where(labels[client_indices] == 2)[0]
        for idx in indices_of_base_images:
            target_idx = np.random.choice(indices_of_target_class)
            base_image = dataset.data[idx]
            target_image = dataset.data[target_idx]
            noisy_image = 0.9 * base_image + 0.1 * target_image
            if dataset_name == 'fmnist':
                noisy_image = noisy_image.to(dataset.data.dtype)
            else:
                noisy_image = noisy_image.astype(np.uint8)
            dataset.data[idx] = noisy_image
            labels[idx] = labels[target_idx]
    dataset.targets = labels
    return dict_users, clients_to_noisy
