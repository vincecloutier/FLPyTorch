import numpy as np

def iid(dataset, num_users):
    """Sample iid client data."""
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

# def noniid(dataset, dataset_name, num_users):
#     """Sample non-iid client data."""
#     if dataset_name == 'mnist' or dataset_name == 'fmnist':
#         num_shards, num_imgs = 600, 100
#     elif dataset_name == 'cifar' or dataset_name == 'resnet':
#         num_shards, num_imgs = 200, 250
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([]) for i in range(num_users)}
#     idxs = np.arange(num_shards*num_imgs)
#     labels = dataset.train_labels.numpy()

#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]

#     # divide and assign 2 shards/client
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
#     return dict_users

def noniid(dataset, dataset_name, num_users, badclient_prop, num_cat):
    """Sample non-IID client data with specified number of categories per client and percentage of non-IID clients."""
    if dataset_name in ['mnist', 'fmnist']:
        shard_size = 100
    elif dataset_name in ['cifar', 'resnet']:
        shard_size = 250

    labels = dataset.train_labels.numpy()
    idxs = np.arange(len(labels))
    num_classes = len(np.unique(labels))

    # create shards per category
    shards_per_category = {}
    shard_id_to_indices = {}
    shard_id = 0
    for c in range(num_classes):
        idxs_c = idxs[labels == c]
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

    if shard_id < num_users * shards_per_client:
        raise ValueError("Not enough shards to assign to all clients")

    # assign shards to iid clients
    assigned_shard_ids = set()
    for i in range(num_iid_clients):
        client_shard_ids = all_shard_ids[i * shards_per_client: (i + 1) * shards_per_client]
        dict_users[i] = np.concatenate([shard_id_to_indices[sid] for sid in client_shard_ids])
        assigned_shard_ids.update(client_shard_ids)

    # assign shards to non iid clients
    for i in range(num_iid_clients, num_users):
        # select k random categories
        categories = np.random.choice(num_classes, num_cat, replace=False)
        available_shards = []
        for c in categories:
            available_shards.extend([sid for sid in shards_per_category[c] if sid not in assigned_shard_ids])

        if len(available_shards) < shards_per_client:
            raise ValueError(f"Not enough shards available for client {i} in selected categories.")

        np.random.shuffle(available_shards)
        client_shard_ids = available_shards[:shards_per_client]
        dict_users[i] = np.concatenate([shard_id_to_indices[sid] for sid in client_shard_ids])
        assigned_shard_ids.update(client_shard_ids)

    return dict_users


def mislabeled(dataset, dict_users, badclient_prop, mislabel_prop):
    """Randomly select a proportion of clients and mislabel a proportion of their samples."""
    labels = dataset.targets.clone()
    for client_id in get_bad_client_indexes(badclient_prop, len(dict_users)):
        client_indices = np.array(list(dict_users[client_id]), dtype=int)
        num_samples = len(client_indices)
        indices_to_mislabel = np.random.choice(client_indices, int(mislabel_prop * num_samples), replace=False)
        for idx in indices_to_mislabel:
            correct_label = labels[idx].item()
            incorrect_labels = list(range(10))
            incorrect_labels.remove(correct_label)
            new_label = np.random.choice(incorrect_labels)
            labels[idx] = new_label
    dataset.targets = labels
    return dict_users


def get_bad_client_indexes(bad_client_prop, num_users):
    """Returns the indexes of all non-IID (or bad) clients."""
    bad_client_indexes = list(range(num_users - int(bad_client_prop * num_users), num_users))
    return bad_client_indexes