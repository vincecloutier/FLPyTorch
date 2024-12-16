import numpy as np

def iid(dataset, num_users):
    """Sample iid client data."""
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def noniid(dataset, dataset_name, num_users, badclient_prop, num_cat):
    """Assigns data to clients ensuring each client receives the same number of samples."""
    dict_users = {i: np.array([], dtype=int) for i in range(num_users)}
    non_iid_clients = []

    # Determine the number of non-iid and iid clients
    num_non_iid = int(badclient_prop * num_users)
    num_iid = num_users - num_non_iid

    # get unique classes
    classes = np.unique(dataset.targets)

    # create a mapping from class to its indices
    class_to_indices = {cls: np.where(dataset.targets == cls)[0].tolist() for cls in classes}
    min_size_of_classes = min([len(class_to_indices[cls]) for cls in classes])
    samples_per_class_per_client = min_size_of_classes // num_users

    # assign an equal number of samples from each class to every client
    for cls in classes:
        cls_indices = class_to_indices[cls]
        np.random.shuffle(cls_indices)
        for i in range(num_users):
            start_idx = i * samples_per_class_per_client
            end_idx = (i + 1) * samples_per_class_per_client
            dict_users[i] = np.concatenate([dict_users[i], cls_indices[start_idx:end_idx]])

    # randomly select iid and non-iid clients
    all_clients = np.arange(num_users)
    iid_clients = np.random.choice(all_clients, num_iid, replace=False)
    non_iid_clients = [i for i in all_clients if i not in iid_clients]

    # remove classes from non-iid clients
    selected_classes = np.random.choice(classes, num_cat, replace=False)
    for client_id in non_iid_clients:
        samples_to_keep = [idx for idx in dict_users[client_id] if dataset.targets[idx] in selected_classes]
        dict_users[client_id] = samples_to_keep

    # for iid clients, remove the same amount of samples except we remove an equal amount all classes
    num_to_remove_per_class = (samples_per_class_per_client * (len(classes) - num_cat)) // len(classes)
    for client_id in iid_clients:
        for cls in classes:
            cls_indices = [idx for idx in dict_users[client_id] if dataset.targets[idx] == cls]
            indices_to_remove = np.random.choice(cls_indices, size=num_to_remove_per_class, replace=False)
            dict_users[client_id] = np.array([idx for idx in dict_users[client_id] if idx not in indices_to_remove])

    return dict_users, non_iid_clients


def mislabeled(dataset, dataset_name, dict_users, badclient_prop, mislabel_prop):
    """Randomly select a proportion of clients and mislabel a proportion of their samples."""
    labels = np.array(dataset.targets)
    clients_to_mislabel = np.random.choice(range(len(dict_users)), int(badclient_prop * len(dict_users)), replace=False)
    num_classes = len(np.unique(labels))
    for client_id in clients_to_mislabel:
        client_indices = np.array(list(dict_users[client_id]), dtype=int)
        indices_to_mislabel = np.random.choice(client_indices, min(len(client_indices), int(mislabel_prop * len(client_indices))), replace=False)
        for idx in indices_to_mislabel:
            correct_label = labels[idx]
            incorrect_labels = list(range(num_classes))
            incorrect_labels.remove(correct_label)
            new_label = np.random.choice(incorrect_labels)
            labels[idx] = new_label
    dataset.targets = labels
    return dict_users, clients_to_mislabel


def noisy(dataset, dataset_name, dict_users, badclient_prop, noisy_proportion):
    """Randomly select a proportion of clients and add noise to a proportion of their samples."""
    l = 0.1 if dataset_name == "fmnist" else 0.9
    c = 7 if dataset_name == "fmnist" else 2
    print(l)
    labels = dataset.targets 
    data = dataset.data 
    clients_to_noisy = np.random.choice(range(len(dict_users)), int(badclient_prop * len(dict_users)), replace=False)
    for client_id in clients_to_noisy:
        client_indices = list(dict_users[client_id])
        indices_of_base_images = [idx for idx in client_indices if labels[idx] != c]
        selected_indices = np.random.choice(indices_of_base_images, min(int(noisy_proportion * len(client_indices)), len(indices_of_base_images)), replace=False)
        indices_of_target_class = np.where(labels == c)[0]
        for idx in selected_indices:
            target_idx = np.random.choice(indices_of_target_class)
            base_image = data[idx]
            target_image = data[target_idx]
            noisy_image = l * base_image + (1-l) * target_image
            data[idx] = noisy_image
            labels[idx] = labels[target_idx]
    dataset.targets = labels
    dataset.data = data
    return dict_users, clients_to_noisy