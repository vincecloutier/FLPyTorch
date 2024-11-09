import numpy as np

def iid(dataset, num_users):
    """Sample iid client data."""
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def noniid(dataset, dataset_name, num_users):
    """Sample non-iid client data."""
    # 60,000 training imgs --> 600 imgs/shard X 100 shards
    if dataset_name == 'mnist':
        num_shards, num_imgs = 600, 100
    elif dataset_name == 'cifar':
        num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def mislabeled(dataset, dataset_name, num_users):
    pass

def noisy(dataset, dataset_name, num_users):
    pass



# TODO:
# 1. for the mislabeled data, we randomly select a proportion of samples from them, and replace their
# labels with random incorrect labels from the same dataset (9-40% of the training set)
# 2. for the noisy data (poison samples), we use a weighted combination of the base image b and the target image t with target opacity p 
# to create the noisy image n = p * b + (1 - p) * t and annotate it with the label of the target image t
# for MNIST we used the weighted combination of an image "6" and target images "2" and for CIFAR10 we used the weighted
# combination of an image "frog" and target images "bird" (9-40% of the training set)
# also note the target opacity is either 0.8 or 0.9     
# 3. we partitioned the fmnist/cmnist datasets (normal, mislabeled, noisy) over 50 clients in both IID and non-IID settings
# we divided images of sorted digits into 600 shards of size 100 and assigned each client 12 shards
# when the 12 shards contains images of 10 different digits we called this the iID 
# 4. we partitioned the cifar10 datasets (normal, mislabeled, noisy) over 50 clients in both IID and non-IID settings
# we divided the dataset into 500 shards of size 100 and assigned each client 10 shards
# when the 10 shards contains images of 10 different classes we called this the iID 



# also note we randomly select 0.6 of the clients in each round