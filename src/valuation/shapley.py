import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import numpy as np
from update import test_inference
from utils import average_weights, initialize_model, get_device
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import torch


# _global_weights = None
# _client_weights = None
# _test_dataset = None


# def init_process(global_weights, client_weights, test_dataset):
#     global _global_weights, _client_weights, _test_dataset
#     _global_weights = global_weights
#     _client_weights = client_weights
#     _test_dataset = test_dataset


# def compute_shapley(args, global_weights, client_weights, test_dataset):
#     """Estimate Shapley values for participants in a round using permutation sampling."""
#     device = get_device()

#     # initialize model and compute base accuracy
#     model = initialize_model(args)
#     model.load_state_dict(global_weights)
#     model.to(device)

#     with torch.no_grad():
#         base_acc = test_inference(model, test_dataset)[0]

#     client_keys = list(client_weights.keys())
#     m = len(client_keys)
#     epsilon, delta, r = 0.25, 0.25, 1  # allow 25% error at 75% confidence
#     t = int((2 * r**2 / epsilon**2) * np.log(2 * m / delta))

#     shapley_updates = defaultdict(float)
  
#     num_processes = args.shapley_processes

#     # avoid passing large data structures to each process
#     pool = multiprocessing.Pool(processes=num_processes, initializer=init_process, initargs=(global_weights, client_weights, test_dataset))

#     args_list = [(client_keys, base_acc, device, args) for _ in range(num_processes)]

#     # progress bar for the overall computation
#     with tqdm(total=t, desc="Computing Shapley Values") as pbar:
#         shapley_update_local = pool.imap_unordered(compute_shapley_for_permutation, args_list)
#         for updates in shapley_update_local:
#             for k, v in updates.items():
#                 shapley_updates[k] += v
#             pbar.update()

#     pool.close()
#     pool.join()

#     del shapley_update_local
#     torch.cuda.empty_cache()

#     # average the values over all permutations
#     shapley_updates = {k: v / t for k, v in shapley_updates.items()}
#     return shapley_updates



def compute_shapley(args, global_weights, client_weights, test_dataset):
    """Estimate Shapley values for participants in a round using permutation sampling."""
    device = get_device()

    # initialize model and compute base accuracy
    model = initialize_model(args)
    model.load_state_dict(global_weights)
    model.to(device)

    with torch.no_grad():
        base_acc = test_inference(model, test_dataset)[0]

    client_keys = list(client_weights.keys())
    m = len(client_keys)
    epsilon, delta, r = 0.25, 0.25, 1  # allow 25% error at 75% confidence
    t = int((2 * r**2 / epsilon**2) * np.log(2 * m / delta))

    shapley_updates = defaultdict(float)

    # prepare arguments for parallel execution
    args_list = [(client_keys, client_weights, global_weights, base_acc, test_dataset, device, args) for _ in range(t)]

    with ProcessPoolExecutor(max_workers=args.shapley_processes) as executor:
        futures = [executor.submit(compute_shapley_for_permutation, arg) for arg in args_list]

        # use tqdm to display progress
        for future in tqdm(as_completed(futures), total=t, desc="Calculating Shapley Values"):
            shapley_update_local = future.result()
            for k, v in shapley_update_local.items():
                shapley_updates[k] += v

    del futures, shapley_update_local
    torch.cuda.empty_cache()

    # average the shapley values over all permutations
    shapley_updates = {k: v / t for k, v in shapley_updates.items()}
    return shapley_updates

def compute_shapley_for_permutation(args):
    # (client_keys, base_acc, device, args_model) = args
    (client_keys, client_weights, global_weights, base_acc, test_dataset, device, args_model) = args

    model = initialize_model(args_model)
    model.load_state_dict(global_weights)
    model.to(device)

    shapley_updates_local = defaultdict(float)

    permutation = np.random.permutation(client_keys)
    prev_acc = base_acc
    current_weights = []
    for i in permutation:
        current_weights.append(client_weights[i])
        avg_weights = average_weights(current_weights)
        model.load_state_dict(avg_weights)
        with torch.no_grad():
            curr_acc = test_inference(model, test_dataset)[0]
        shapley_updates_local[i] += curr_acc - prev_acc
        prev_acc = curr_acc

    del model
    torch.cuda.empty_cache()

    return shapley_updates_local

