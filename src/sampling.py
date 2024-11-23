import os
import json
import subprocess
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


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
    if args.dataset in ['cifar', 'resnet', 'mobilenet']:
        data_dir = './data/cifar/'
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        full_train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)
    elif args.dataset == 'fmnist':
        data_dir = './data/fmnist/'
        train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        full_train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=test_transform)
    else:
        data_dir = './data/imagenet/'
        train_transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        download_imagenet(data_dir) 
        full_train_dataset = datasets.ImageNet(data_dir, split='train', transform=train_transform)
        test_dataset = datasets.ImageNet(data_dir, split='val', transform=test_transform)

    # split the training data into training and validation sets
    train_dataset, valid_dataset = train_val_split(full_train_dataset, val_prop=0.1)

    if args.setting == 0:
        user_groups, bad_clients = iid(train_dataset, args.num_users), None
    elif args.setting == 1:
        user_groups, bad_clients = noniid(train_dataset, args.dataset, args.num_users, args.badclient_prop, args.num_categories_per_client)
    elif args.setting == 2:
        iid_user_groups = iid(train_dataset, args.num_users)
        user_groups, bad_clients = mislabeled(train_dataset, args.dataset, iid_user_groups, args.badclient_prop, args.badsample_prop)
    elif args.setting == 3:
        iid_user_groups = iid(train_dataset, args.num_users)
        user_groups, bad_clients = noisy(train_dataset, args.dataset, iid_user_groups, args.badclient_prop, args.badsample_prop)
    else:
        raise ValueError("Invalid value for --setting. Please use 0, 1, 2, or 3.")

    return train_dataset, valid_dataset, test_dataset, user_groups, bad_clients

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
    subprocess.run(["kaggle", "datasets", "download", "-d", "imagenet-object-localization-challenge", "-p", data_dir, "--force"], check=True)


def train_val_split(full_train_dataset, val_prop):
    num_train = len(full_train_dataset)
    split = int(np.floor(val_prop * num_train))
    indices = list(range(num_train))
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    return SubsetSplit(full_train_dataset, train_idx), SubsetSplit(full_train_dataset, valid_idx)


def iid(dataset, num_users):
    num_items = len(dataset) // num_users
    dict_users, all_idxs = {}, list(range(len(dataset)))
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def noniid(dataset, dataset_name, num_users, badclient_prop, num_cat):
    shard_size = 100 if dataset_name == 'fmnist' else 250
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
    shards_per_client = len(all_shard_ids) // num_users
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
        available_shards = [sid for c in categories for sid in shards_per_category[c]  if sid not in assigned_shard_ids]
        np.random.shuffle(available_shards)
        client_shard_ids = available_shards[:shards_per_client]
        dict_users[i] = np.concatenate([shard_id_to_indices[sid] for sid in client_shard_ids])
        assigned_shard_ids.update(client_shard_ids)

    return dict_users, non_iid_clients


def mislabeled(dataset, dataset_name, dict_users, badclient_prop, mislabel_prop):
    labels = np.array(dataset.targets)
    num_clients = len(dict_users)
    num_mislabel_clients = int(badclient_prop * num_clients)
    clients_to_mislabel = np.random.choice(list(dict_users.keys()), num_mislabel_clients, replace=False)
    num_classes = len(np.unique(labels))
    for client_id in clients_to_mislabel:
        client_indices = list(dict_users[client_id])
        num_mislabels = int(mislabel_prop * len(client_indices))
        indices_to_mislabel = np.random.choice(client_indices, num_mislabels, replace=False)
        for idx in indices_to_mislabel:
            correct_label = labels[idx]
            incorrect_labels = list(range(num_classes))
            incorrect_labels.remove(correct_label)
            new_label = np.random.choice(incorrect_labels)
            labels[idx] = new_label
    dataset.targets = labels.tolist()
    return dict_users, list(clients_to_mislabel)


def noisy(dataset, dataset_name, dict_users, badclient_prop, noisy_proportion):
    labels = np.array(dataset.targets)
    data = dataset.data.copy()
    num_clients = len(dict_users)
    num_noisy_clients = int(badclient_prop * num_clients)
    clients_to_noisy = np.random.choice(list(dict_users.keys()), 
                                       num_noisy_clients, replace=False)
    for client_id in clients_to_noisy:
        client_indices = list(dict_users[client_id])
        num_noisy = int(noisy_proportion * len(client_indices))
        indices_to_noise = np.random.choice(client_indices, num_noisy, replace=False)
        target_class = 2  # This can be parameterized if needed
        base_indices = [idx for idx in indices_to_noise if labels[idx] != target_class]
        if not base_indices:
            continue
        target_indices = np.where(labels == target_class)[0]
        for idx in base_indices:
            if len(target_indices) == 0:
                continue
            target_idx = np.random.choice(target_indices)
            noisy_image = 0.9 * data[idx].astype(float) + 0.1 * data[target_idx].astype(float)
            if dataset_name == 'fmnist':
                noisy_image = noisy_image.astype(dataset.data.dtype)
            else:
                noisy_image = noisy_image.astype(np.uint8)
            data[idx] = noisy_image
            labels[idx] = labels[target_idx]
    dataset.data = data
    dataset.targets = labels.tolist()
    return dict_users, list(clients_to_noisy)
