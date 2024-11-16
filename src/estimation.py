import torch
from utils import get_device
from update import compute_hessian

def compute_bv_simple(args, gradient, delta_t_i):
    """Computes the simplified banzhaf value for client i at epoch t."""
    # compute delta term
    delta_term = {key: (-1.0 / args.num_users) * delta_t_i[key] for key in delta_t_i}

    # compute gradient dot product delta_term
    bv = 0.0
    for name in delta_term:
        bv += torch.dot(gradient[name].view(-1), delta_term[name].view(-1))
    return bv.item()

def compute_bv_hvp(args, model, train_dataset, gradient, delta_t_i, accumulated_Delta_G_i):
    """Computes the banzhaf value component for client i at epoch t."""
    device = get_device()
    # compute delta term
    delta_term = {key: (-1.0 / args.num_users) * delta_t_i[key] for key in delta_t_i}

    # prepare accumulated_Delta_G_i as a list matching model parameters
    accumulated_Delta_G_i_list = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            accumulated_Delta_G_i_list.append(accumulated_Delta_G_i[name].to(device))

    # compute Hessian-vector product
    hessian_term = compute_hessian(model, train_dataset, accumulated_Delta_G_i_list)

    # multiply by alpha
    # hessian_term = {name: args.alpha * hvp_dict[name] for name in hvp_dict}

    # compute total_term = delta_term + hessian_term
    total_term = {}
    for name in delta_term:
        total_term[name] = delta_term[name] + hessian_term.get(name, torch.zeros_like(delta_term[name], device=device))

    # compute gradient dot product total_term
    final_term = 0.0
    for name in total_term:
        final_term += torch.dot(gradient[name].view(-1), total_term[name].view(-1))
    return final_term.item()

def compute_G_t(delta_t_i_dict, keys):
    """Computes G_t = (1/n) sum_{k=1}^{n} delta_{t,k}"""
    num_clients = len(delta_t_i_dict)
    G_t = {}
    for key in keys:
        delta_sum = sum(delta_t_i_dict[idx][key] for idx in delta_t_i_dict)
        G_t[key] = delta_sum / num_clients
    return G_t

def compute_G_minus_i_t(delta_t_i_dict, keys, idx_to_exclude):
    """Computes G_{-i}^t = (1/(n-1)) sum_{k != i} delta_{t,k}"""
    num_clients = len(delta_t_i_dict) - 1
    if num_clients == 0:
        # only one client, return zeros
        G_minus_i_t = {key: torch.zeros_like(delta_t_i_dict[idx_to_exclude][key]) for key in keys}
        return G_minus_i_t
    G_minus_i_t = {}
    for key in keys:
        delta_sum = sum(delta_t_i_dict[idx][key] for idx in delta_t_i_dict if idx != idx_to_exclude)
        G_minus_i_t[key] = delta_sum / num_clients
    return G_minus_i_t