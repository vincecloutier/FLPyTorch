import itertools
from math import factorial as fact
from collections import defaultdict
import time
import torch
import numpy as np
from options import args_parser
from update import inference, gradient
from torch.utils.data import DataLoader
from estimation import compute_bv_simple, compute_bv_hvp, compute_G_t, compute_G_minus_i_t
from utils import average_weights, setup_logger, get_device, identify_bad_idxs, measure_accuracy, initialize_model
from sampling import get_dataset, SubsetSplit
from scipy.stats import pearsonr
import copy
from torch import nn
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define the ClientTrainer as a standard class
class ClientTrainer:
    def __init__(self, args, trainloader, device):
        self.args = args
        self.trainloader = trainloader
        self.device = device
        print(f"Trainer using GPU: {self.device}")
        self.model = initialize_model(args).to(self.device)
        self.optimizer = self._get_optimizer()
        self.criterion = nn.CrossEntropyLoss()

    def _get_optimizer(self):
        if self.args.optimizer == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        else:
            raise ValueError(f"Unsupported optimizer: {self.args.optimizer}")

    def update_weights(self, global_weights):
        self.model.load_state_dict(global_weights)

    def train(self, global_weights, local_ep):
        self.update_weights(global_weights)
        self.model.train()
        epoch_loss = []
        for _ in range(local_ep):
            batch_loss = []
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                log_probs = self.model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return self.model.state_dict()

# deine the train_subset function without Ray
def train_subset(args, global_weights, client_trainers, valid_dataset, test_dataset, clients):
    model = initialize_model(args).to(device)

    # sort the clients
    subset_key = tuple(sorted(clients))
    isBanzhaf = subset_key == (0, 1, 2, 3, 4)
    best_test_acc, best_test_loss = 0, float('inf')
    approx_banzhaf_values_hessian = defaultdict(float)
    approx_banzhaf_values_simple = defaultdict(float)
    delta_t = defaultdict(dict) if isBanzhaf else None
    delta_g = defaultdict(lambda: {key: torch.zeros_like(global_weights[key]) for key in global_weights.keys()}) if isBanzhaf else None
    no_improvement_count = 0

    for epoch in tqdm(range(args.epochs), desc=f"Training for subset: {subset_key}"):
        local_weights = []

        # sample a fraction of clients
        m = max(int(args.frac * len(subset_key)), 1)
        idxs_users = np.random.choice(subset_key, m, replace=False)

        
        training_futures = []
        for idx in idxs_users:
            local_weights.append(client_trainers[idx].train(global_weights, args.local_ep))

        # if using banzhaf, compute values
        if isBanzhaf:
            grad = gradient(args, initialize_model(args).to(device), valid_dataset, device)  # Adjusted for example
            G_t = compute_G_t(delta_t[epoch], global_weights.keys())
            for idx in idxs_users:
                G_t_minus_i = compute_G_minus_i_t(delta_t[epoch], global_weights.keys(), idx)
                if epoch > 0:
                    for key in global_weights.keys():
                        delta_g[idx][key] += G_t_minus_i[key] - G_t[key]
                approx_banzhaf_values_hessian[idx] += compute_bv_hvp(args, initialize_model(args).to(device), test_dataset, grad, delta_t[epoch][idx], delta_g[idx], device)
                approx_banzhaf_values_simple[idx] += compute_bv_simple(args, grad, delta_t[epoch][idx])

        # average the local weights to update global weights
        global_weights = average_weights(local_weights)

        # update global weights across all trainers
        for trainer in client_trainers.values():
            trainer.update_weights(global_weights)

        # evaluate the global model using one of the trainers
        test_acc, test_loss = inference(args, initialize_model(args).to(device), test_dataset, device)
        if test_acc > best_test_acc * 1.01 or test_loss < best_test_loss * 0.99:
            best_test_acc = test_acc
            best_test_loss = test_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count > 3:
                break

    torch.cuda.empty_cache()
    return subset_key, best_test_loss, best_test_acc, approx_banzhaf_values_simple, approx_banzhaf_values_hessian

# main execution block
if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    logger = setup_logger(f'benchmark_{args.dataset}_{args.setting}')

    # Load datasets
    train_dataset, valid_dataset, test_dataset, user_groups, actual_bad_clients = get_dataset(args)

    # Initialize global model
    global_model = initialize_model(args)
    global_weights = global_model.state_dict()

    # prepare dataloaders
    train_loaders = {user_id: DataLoader(SubsetSplit(train_dataset, indices), batch_size=args.local_bs, shuffle=True, num_workers=args.num_workers) for user_id, indices in user_groups.items()}

    # check available GPUs
    available_gpus = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [torch.device("cpu")]

    num_trainers = min(5, len(available_gpus))  # limit to available GPUs
    client_trainers = {}
    for trainer_id in range(num_trainers):
        # assign clients to each trainer (simple distribution)
        assigned_clients = [user_id for user_id in user_groups.keys() if user_id % num_trainers == trainer_id]
        # combine their trainloaders
        if assigned_clients:
            combined_dataset = torch.utils.data.ConcatDataset([SubsetSplit(train_dataset, user_groups[user_id]) for user_id in assigned_clients])
            combined_loader = DataLoader(
                combined_dataset,
                batch_size=args.local_bs,
                shuffle=True,
                num_workers=args.num_workers
            )
        else:
            # If no clients are assigned, use an empty dataset
            combined_loader = DataLoader(
                torch.utils.data.TensorDataset(torch.empty(0), torch.empty(0)),
                batch_size=args.local_bs,
                shuffle=True,
                num_workers=args.num_workers
            )
        # assign GPU or CPU
        device = available_gpus[trainer_id] if trainer_id < len(available_gpus) else torch.device("cpu")
        # initialize the trainer
        trainer = ClientTrainer(args, combined_loader, device)
        client_trainers[trainer_id] = trainer

    # broadcast initial global weights to all trainers
    for trainer in client_trainers.values():
        trainer.update_weights(global_weights)

    # references to datasets
    valid_dataset_ref = valid_dataset
    test_dataset_ref = test_dataset

    # generate all possible subsets (consider limiting this if num_users is large)
    all_subsets = list(itertools.chain.from_iterable(itertools.combinations(range(args.num_users), r) for r in range(args.num_users, -1, -1)))

    # launch train_subset tasks in parallel with controlled concurrency
    results_list = []
    with ThreadPoolExecutor(max_workers=args.processes) as executor:
        future_to_subset = {executor.submit(train_subset, args, copy.deepcopy(global_weights), client_trainers, clients, valid_dataset_ref, test_dataset_ref
            ): clients for clients in all_subsets
        }

        for future in tqdm(as_completed(future_to_subset), total=len(future_to_subset), desc="Processing subsets"):
            try:
                result = future.result()
                results_list.append(result)
            except Exception as exc:
                subset = future_to_subset[future]
                print(f'Subset {subset} generated an exception: {exc}')

    results = {subset_key: (loss, accuracy, abv_simple, abv_hessian) for subset_key, loss, accuracy, abv_simple, abv_hessian in results_list}

    # compute shapley and banzhaf values
    shapley_values, banzhaf_values = defaultdict(float), defaultdict(float)
    for client in range(args.num_users):
        for r in range(args.num_users):
            for subset in itertools.combinations([c for c in range(args.num_users) if c != client], r):
                subset_key = tuple(sorted(subset))
                subset_with_client_key = tuple(sorted(subset + (client,)))
                if subset_key in results and subset_with_client_key in results:
                    mc = results[subset_key][0] - results[subset_with_client_key][0]
                    shapley_values[client] += ((fact(len(subset)) * fact(args.num_users - len(subset) - 1)) / fact(args.num_users)) * mc
                    banzhaf_values[client] += mc / len(all_subsets)

    # identify the subset with the longest key (assuming it's the full set)
    longest_client_key = max(results.keys(), key=lambda x: len(x))
    test_loss, test_acc, abv_simple, abv_hessian = results[longest_client_key]

    # identify bad clients using approximate banzhaf values
    identified_bad_clients_simple = identify_bad_idxs(abv_simple)
    identified_bad_clients_hessian = identify_bad_idxs(abv_hessian)
    bad_client_accuracy_simple = measure_accuracy(actual_bad_clients, identified_bad_clients_simple)
    bad_client_accuracy_hessian = measure_accuracy(actual_bad_clients, identified_bad_clients_hessian)

    # prepare data for correlation
    shared_clients = set(shapley_values.keys()) & set(banzhaf_values.keys()) & set(abv_simple.keys()) & set(abv_hessian.keys())
    sv = [shapley_values[client] for client in shared_clients]
    bv = [banzhaf_values[client] for client in shared_clients]
    abv_simple_list = [abv_simple[client] for client in shared_clients]
    abv_hessian_list = [abv_hessian[client] for client in shared_clients]

    # logging results
    setting_str = {
        0: "IID",
        1: f"Non IID with {len(actual_bad_clients)} Bad Clients and {args.num_categories_per_client} Categories Per Bad Client",
        2: f"Mislabeled with {len(actual_bad_clients)} Bad Clients and {100 * args.badsample_prop}% Bad Samples Per Bad Client",
        3: f"Noisy with {len(actual_bad_clients)} Bad Clients and {100 * args.badsample_prop}% Bad Samples Per Bad Client"
    }.get(args.setting, "Unknown Setting")
    logger.info(f'Number Of Clients: {args.num_users}, Client Selection Fraction: {args.frac}, Local Epochs: {args.local_ep}, Batch Size: {args.local_bs}')
    logger.info(f'Dataset: {args.dataset}, Setting: {setting_str}, Number Of Rounds: {args.epochs}')
    logger.info(f'Test Accuracy Of Global Model: {100 * test_acc}%')
    logger.info(f'Shapley Values: {sv}')
    logger.info(f'Banzhaf Values: {bv}')
    logger.info(f'Approximate Banzhaf Values Simple: {abv_simple_list}')
    logger.info(f'Approximate Banzhaf Values Hessian: {abv_hessian_list}')
    logger.info(f'Pearson Correlation Between Shapley And Banzhaf Values: {pearsonr(sv, bv)}')
    logger.info(f'Pearson Correlation Between Shapley And Approximate Banzhaf Values Simple: {pearsonr(sv, abv_simple_list)}')
    logger.info(f'Pearson Correlation Between Shapley And Approximate Banzhaf Values Hessian: {pearsonr(sv, abv_hessian_list)}')
    logger.info(f'Pearson Correlation Between Banzhaf And Approximate Banzhaf Values Simple: {pearsonr(bv, abv_simple_list)}')
    logger.info(f'Pearson Correlation Between Banzhaf And Approximate Banzhaf Values Hessian: {pearsonr(bv, abv_hessian_list)}')
    logger.info(f'Actual Bad Clients: {actual_bad_clients}')
    logger.info(f'Identified Bad Clients Simple: {identified_bad_clients_simple}')
    logger.info(f'Identified Bad Clients Hessian: {identified_bad_clients_hessian}')
    logger.info(f'Bad Client Accuracy Simple: {bad_client_accuracy_simple}')
    logger.info(f'Bad Client Accuracy Hessian: {bad_client_accuracy_hessian}')
    logger.info(f'Total Run Time: {time.time() - start_time} seconds')