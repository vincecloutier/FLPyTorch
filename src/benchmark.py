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
import ray
import copy
from torch import nn
from tqdm import tqdm  
import ray.train.torch
from concurrent.futures import ThreadPoolExecutor

# Initialize Ray at the start
ray.init(include_dashboard=True, logging_level="INFO", object_store_memory=20 * 1024**3, num_gpus=3, num_cpus=36)

# Define a Ray actor for client training
@ray.remote
class ClientTrainer:
    def __init__(self, args, trainloader, device):
        self.args = args
        self.trainloader = trainloader
        self.device = device
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

# Define the train_subset function without nested remote calls
@ray.remote
def train_subset(args, global_weights, client_trainers, subset_key, isBanzhaf, valid_dataset_ref, test_dataset_ref):
    device = get_device()

    if not subset_key:
        return subset_key, float('inf'), 0, defaultdict(float), defaultdict(float)

    # Initialize local variables
    best_test_acc, best_test_loss = 0, float('inf')
    approx_banzhaf_values_hessian = defaultdict(float)
    approx_banzhaf_values_simple = defaultdict(float)
    delta_t = defaultdict(dict) if isBanzhaf else None
    delta_g = defaultdict(lambda: {key: torch.zeros_like(global_weights[key]) for key in global_weights.keys()}) if isBanzhaf else None

    no_improvement_count = 0
    epochs = args.epochs

    for epoch in tqdm(range(epochs), desc=f"Training for subset: {subset_key}"):
        local_weights = []
        
        # Sample a fraction of clients
        m = max(int(args.frac * len(subset_key)), 1)
        idxs_users = np.random.choice(subset_key, m, replace=False)

        # Parallel training using ThreadPoolExecutor within the remote task
        with ThreadPoolExecutor(max_workers=m) as executor:
            futures = [executor.submit(client_trainers[idx].train.remote, global_weights, args.local_ep) for idx in idxs_users]
            client_results = ray.get([f.result() for f in futures])

        for idx, w in zip(idxs_users, client_results):
            local_weights.append(copy.deepcopy(w))
            if isBanzhaf:
                delta_t[epoch][idx] = {key: (global_weights[key] - w[key]) for key in w.keys()}

        if isBanzhaf:
            # Compute Banzhaf values
            grad = gradient(args, client_trainers[0].model, ray.get(valid_dataset_ref), device)  # Assuming gradient is same across clients
            G_t = compute_G_t(delta_t[epoch], global_weights.keys())
            for idx in idxs_users:
                G_t_minus_i = compute_G_minus_i_t(delta_t[epoch], global_weights.keys(), idx)
                if epoch > 0:
                    for key in global_weights.keys():
                        delta_g[idx][key] += G_t_minus_i[key] - G_t[key]
                approx_banzhaf_values_hessian[idx] += compute_bv_hvp(args, client_trainers[0].model, ray.get(test_dataset_ref), grad, delta_t[epoch][idx], delta_g[idx], device)
                approx_banzhaf_values_simple[idx] += compute_bv_simple(args, grad, delta_t[epoch][idx])

        # Average the local weights to update global weights
        global_weights = average_weights(local_weights)

        # Update global weights across all clients
        update_futures = [trainer.update_weights.remote(global_weights) for trainer in client_trainers.values()]
        ray.get(update_futures)

        # Evaluate the global model
        test_acc, test_loss = inference(args, client_trainers[0].model, ray.get(test_dataset_ref), device)  # Assuming inference can be done on any client model
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

# Main execution block
if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    logger = setup_logger(f'benchmark_{args.dataset}_{args.setting}')

    # Load datasets
    train_dataset, valid_dataset, test_dataset, user_groups, actual_bad_clients = get_dataset(args)

    # Initialize global model
    global_model = initialize_model(args)
    global_weights = global_model.state_dict()

    # Prepare DataLoaders
    train_loaders = {
        user_id: DataLoader(
            SubsetSplit(train_dataset, indices),
            batch_size=args.local_bs,
            shuffle=True,
            num_workers=args.num_workers
        )
        for user_id, indices in user_groups.items()
    }

    # Create Ray actors for each client
    client_trainers = {
        user_id: ClientTrainer.remote(args, ray.put(train_loaders[user_id]), get_device())
        for user_id in user_groups.keys()
    }

    # Broadcast initial global weights to all clients
    update_futures = [trainer.update_weights.remote(global_weights) for trainer in client_trainers.values()]
    ray.get(update_futures)

    # References to datasets to avoid data copying
    valid_dataset_ref = ray.put(valid_dataset)
    test_dataset_ref = ray.put(test_dataset)

    all_subsets = list(itertools.chain.from_iterable(itertools.combinations(range(args.num_users), r) for r in range(args.num_users, -1, -1)))

    subset_info = {
        subset: (subset == (0, 1, 2, 3, 4))  # Modify this condition based on your specific Banzhaf subset criteria
        for subset in all_subsets
    }

    futures = [train_subset.remote(args, copy.deepcopy(global_weights), client_trainers, clients, isBanzhaf) for clients, isBanzhaf in subset_info.items()]

    results_list = ray.get(futures)
    results = {
        subset_key: (loss, accuracy, abv_simple, abv_hessian)
        for subset_key, loss, accuracy, abv_simple, abv_hessian in results_list
    }

    shapley_values, banzhaf_values = defaultdict(float), defaultdict(float)
    for client in range(args.num_users):
        for r in range(args.num_users):
            for subset in itertools.combinations([c for c in range(args.num_users) if c != client], r):
                subset_key = tuple(sorted(subset))
                subset_with_client_key = tuple(sorted(subset + (client,)))
                mc = results[subset_key][0] - results[subset_with_client_key][0]
                shapley_values[client] += ((fact(len(subset)) * fact(args.num_users - len(subset) - 1)) / fact(args.num_users)) * mc
                banzhaf_values[client] += mc / len(all_subsets)

    longest_client_key = max(results.keys(), key=len)
    test_loss, test_acc, abv_simple, abv_hessian = results[longest_client_key]

    identified_bad_clients_simple = identify_bad_idxs(abv_simple)
    identified_bad_clients_hessian = identify_bad_idxs(abv_hessian)
    bad_client_accuracy_simple = measure_accuracy(actual_bad_clients, identified_bad_clients_simple)
    bad_client_accuracy_hessian = measure_accuracy(actual_bad_clients, identified_bad_clients_hessian)

    shared_clients = set(shapley_values.keys()) & set(banzhaf_values.keys()) & set(abv_simple.keys()) & set(abv_hessian.keys())
    sv = [shapley_values[client] for client in shared_clients]
    bv = [banzhaf_values[client] for client in shared_clients]
    abv_simple = [abv_simple[client] for client in shared_clients]
    abv_hessian = [abv_hessian[client] for client in shared_clients]

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
    logger.info(f'Approximate Banzhaf Values Simple: {abv_simple}')
    logger.info(f'Approximate Banzhaf Values Hessian: {abv_hessian}')
    logger.info(f'Pearson Correlation Between Shapley And Banzhaf Values: {pearsonr(sv, bv)}')
    logger.info(f'Pearson Correlation Between Shapley And Approximate Banzhaf Values Simple: {pearsonr(sv, abv_simple)}')
    logger.info(f'Pearson Correlation Between Shapley And Approximate Banzhaf Values Hessian: {pearsonr(sv, abv_hessian)}')
    logger.info(f'Pearson Correlation Between Banzhaf And Approximate Banzhaf Values Simple: {pearsonr(bv, abv_simple)}')
    logger.info(f'Pearson Correlation Between Banzhaf And Approximate Banzhaf Values Hessian: {pearsonr(bv, abv_hessian)}')
    logger.info(f'Actual Bad Clients: {actual_bad_clients}')
    logger.info(f'Identified Bad Clients Simple: {identified_bad_clients_simple}')
    logger.info(f'Identified Bad Clients Hessian: {identified_bad_clients_hessian}')
    logger.info(f'Bad Client Accuracy Simple: {bad_client_accuracy_simple}')
    logger.info(f'Bad Client Accuracy Hessian: {bad_client_accuracy_hessian}')
    logger.info(f'Total Run Time: {time.time() - start_time} seconds')
    ray.shutdown()