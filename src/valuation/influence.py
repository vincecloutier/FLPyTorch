import torch
from tqdm import tqdm
from collections import defaultdict
from update import LocalUpdate, test_gradient, conjugate_gradient
import multiprocessing
from functools import partial


def compute_influence_functions(args, model, train_dataset, user_groups, device, test_dataset):
    """Compute Influence Functions for clients."""
    influence_values = defaultdict(float)
    model.eval()
    
    # compute the gradient of the test loss w.r.t. model parameters
    test_loss_grad = test_gradient(args, model, test_dataset)
    
    # solve Hx = grad_test_loss to get x = H^{-1} grad_test_loss
    x = conjugate_gradient(model, train_dataset, test_loss_grad, num_iterations=20, tol=1e-4)
    
    # split x back into parameter tensors
    x_dict = {}
    idx = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            numel = param.numel()
            x_dict[name] = x[idx:idx+numel].view_as(param).clone().detach()
            idx += numel

    # parallelize the computation of influence for each client
    with multiprocessing.Pool(processes=args.processes) as pool:
        results = list(tqdm(pool.imap(
            partial(compute_client_influence, args=args, model=model, train_dataset=train_dataset, 
                    user_groups=user_groups, x=x, device=device),
            range(args.num_users)
        ), total=args.num_users, desc="Influence Functions"))
    
    for client_idx, influence in results:
        influence_values[client_idx] = influence
    
    return influence_values

def compute_client_influence(client_idx, args, model, train_dataset, user_groups, x):
    """Compute the influence of a single client."""
    client_data = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[client_idx]).get_data()
    client_grad = test_gradient(args, model, client_data)
    client_grad_flat = torch.cat([g.contiguous().view(-1) for g in client_grad])
    influence = -torch.dot(client_grad_flat, x).item()
    return client_idx, influence
    