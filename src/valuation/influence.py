import copy
from collections import defaultdict
from typing import Tuple
import torch
import torch.nn.functional as F
from torch import nn
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.task import Task
from kronfluence.utils.dataset import DataLoaderKwargs
from utils import get_device, initialize_model

BATCH_TYPE = Tuple[torch.Tensor, torch.Tensor]




def compute_influence(args, global_weights, train_dataset, test_dataset, user_groups):
    # Unpack arguments
    loss_fn = F.cross_entropy
    device = get_device()
    lambda_reg = 1e-2

    model = initialize_model(args).to(device)
    model.load_state_dict(global_weights)
    model.eval()

    # Extract parameters
    theta = list(model.parameters())
    for p in theta:
        p.requires_grad = True

    ###########################################################################
    # Helper functions
    ###########################################################################
    def compute_grad_and_hessian(x, y, model, loss_fn):
        model.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        grads = torch.autograd.grad(loss, theta, create_graph=True)
        grad_vec = torch.cat([g.reshape(-1) for g in grads])
        
        # compute Hessian (naive O(d^2))
        dim = grad_vec.numel()
        hessian_rows = []
        for i in range(dim):
            # take gradient of grad_vec[i] w.r.t. parameters
            grad2 = torch.autograd.grad(grad_vec[i], theta, retain_graph=True)
            grad2_vec = torch.cat([g.reshape(-1) for g in grad2])
            hessian_rows.append(grad2_vec)
        hessian = torch.stack(hessian_rows, dim=0)
        return grad_vec.detach(), hessian.detach()

    def compute_f_gradient(model, loss_fn, test_dataset):
        # compute gradient of f(θ) = average validation loss w.r.t. θ
        model.zero_grad()
        total_loss = 0.0
        for idx in range(len(test_dataset)):
            xv, yv = test_dataset[idx]
            xv = xv.unsqueeze(0).to(device)
            yv = yv.to(device)
            outv = model(xv)
            loss_v = loss_fn(outv, yv)
            total_loss += loss_v
        test_loss = total_loss / len(test_dataset)

        grad_f = torch.autograd.grad(test_loss, theta)
        grad_f_vec = torch.cat([g.reshape(-1) for g in grad_f])
        return grad_f_vec.detach()

    ###########################################################################
    # Compute Hessians for each user/client and average
    ###########################################################################
    # We will also store all sample gradients for later use
    all_sample_grads = []
    hessians = []

    with torch.no_grad():
        # Temporarily disable no_grad to allow Hessian computation
        pass

    # Actually we need gradients, so re-enable grad for Hessians:
    torch.set_grad_enabled(True)

    for user_id, sample_indices in user_groups.items():
        hess_k_list = []
        n_k = len(sample_indices)

        for idx in sample_indices:
            x, y = train_dataset[idx]
            x = x.unsqueeze(0).to(device)
            y = y.to(device)

            # Compute gradient and hessian for this single sample
            grad_vec, hess = compute_grad_and_hessian(x, y, model, loss_fn)
            all_sample_grads.append(grad_vec)
            hess_k_list.append(hess)

        # Average Hessian for this client
        H_k = sum(hess_k_list) / n_k
        hessians.append(H_k)

    # Average Hessian across clients
    H_avg = sum(hessians) / len(hessians)

    # Add lambda I for stability
    dim = H_avg.size(0)
    H_reg = H_avg + lambda_reg * torch.eye(dim, device=device)

    # invert Hessian-like matrix
    H_inv = torch.inverse(H_reg)

    ###########################################################################
    # Suppose we want to compute the influence of removing a particular set of samples.
    # Let's say we pick a particular client and some subset of their samples as w_k.
    # For a real scenario, you'd pass w_k or compute it based on your needs.
        #
    # Here, we just demonstrate by considering removing all samples from a single user or
    # from a given mask. Adjust as needed.
    ###########################################################################
    # Example: consider the first user's samples as w_k
    first_user_id = list(user_groups.keys())[0]
    w_k_indices = user_groups[first_user_id]  # the samples we "remove"
    w_k_mask = torch.zeros(len(all_sample_grads), dtype=torch.bool, device=device)
    # all_sample_grads currently holds grads in order of iteration over all users
    # If you need a direct mapping, you should store user and index info. 
    # For simplicity, assume all_sample_grads follows the order of user_groups:
    count = 0
    user_boundaries = {}
    for uid, inds in user_groups.items():
        user_boundaries[uid] = (count, count + len(inds))
        count += len(inds)

    start_idx, end_idx = user_boundaries[first_user_id]
    w_k_mask[start_idx:end_idx] = True

    # stack all sample grads
    all_grads_matrix = torch.stack(all_sample_grads, dim=0)
    # compute g_{θ,f}(w_k)
    # the definition might differ, but let's assume sum or mean. We use sum here:
    g_theta_f_wk = all_grads_matrix[w_k_mask].mean(dim=0)

    # compute ∇_θ f(θ̂(1))
    grad_f_theta = compute_f_gradient(model, loss_fn, test_dataset)

    # compute influence: I_f(w_k) = grad_f_theta^T * H_inv * g_theta_f_wk
    influence = grad_f_theta @ H_inv @ g_theta_f_wk

    return influence.item()


class ClassificationTask(Task):
    def compute_train_loss(self, batch: BATCH_TYPE, model: nn.Module, sample: bool = False) -> torch.Tensor:
        inputs, labels = batch
        logits = model(inputs)
        if not sample:
            return F.cross_entropy(logits, labels, reduction="sum")
        with torch.no_grad():
            probs = torch.nn.functional.softmax(logits.detach(), dim=-1)
            sampled_labels = torch.multinomial(probs, num_samples=1).flatten()
        return F.cross_entropy(logits, sampled_labels, reduction="sum")

    def compute_measurement(self, batch: BATCH_TYPE, model: nn.Module) -> torch.Tensor:
        inputs, labels = batch
        logits = model(inputs)

        bindex = torch.arange(logits.shape[0]).to(device=logits.device, non_blocking=False)
        logits_correct = logits[bindex, labels]

        cloned_logits = logits.clone()
        cloned_logits[bindex, labels] = torch.tensor(-torch.inf, device=logits.device, dtype=logits.dtype)

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return -margins.sum()


def compute_influence(args, global_weights, train_dataset, user_groups, noise_transform):
    device = get_device()

    # applying noise transform to train_dataset
    t_dataset = copy.deepcopy(train_dataset)
    noise_transform.to('cpu')
    t_dataset.data = [noise_transform(torch.tensor(data, dtype=torch.float32)) for data in t_dataset.data]
    noise_transform.to(device)

    # prepare the model
    model = initialize_model(args)
    model.load_state_dict(global_weights)
    model.to(device).float()
    model.eval()

    # define task and prepare model
    task = ClassificationTask()
    model = prepare_model(model, task)
    analyzer = Analyzer(analysis_name="analysis", model=model, task=task, profile=False)

    # configure parameters
    dataloader_kwargs = DataLoaderKwargs(num_workers=4)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    # compute influence factors
    factor_args = FactorArguments(
        # TODO: test settings for strategy
        strategy=args.strategy,
        use_empirical_fisher=True,
        amp_dtype=torch.bfloat16,
        amp_scale=2.0**16,

        # settings for fitting covariance and lambda matrix
        # covariance_max_examples=5000,
        # lambda_max_examples=5000,

        # precision settings
        eigendecomposition_dtype=torch.float32,
        activation_covariance_dtype=torch.bfloat16,
        gradient_covariance_dtype=torch.bfloat16,
        per_sample_gradient_dtype=torch.bfloat16,
        lambda_dtype=torch.bfloat16,
    )
    analyzer.fit_all_factors(
        factors_name=args.strategy,
        dataset=t_dataset,
        per_device_batch_size=None,
        factor_args=factor_args,
        overwrite_output_dir=True,
    )

    # compute influence scores
    score_args = ScoreArguments(
        # TODO: test settings for damping
        damping_factor=1e-2,
        amp_dtype=torch.bfloat16,
        
        # TODO: test if you should use this w/ old measurement?
        use_measurement_for_self_influence=False,

        # precision settings
        query_gradient_svd_dtype=torch.bfloat16,
        per_sample_gradient_dtype=torch.bfloat16,
        precondition_dtype=torch.bfloat16,
        score_dtype=torch.bfloat16
    )

    analyzer.compute_self_scores(
        scores_name=args.strategy,
        factors_name=args.strategy,
        train_dataset=t_dataset,
        score_args=score_args,
        overwrite_output_dir=True,
    )

    # load scores
    scores = analyzer.load_self_scores(args.strategy)["all_modules"]
    print(f"Scores shape: {scores.shape}")
    client_influences = defaultdict(float)
    for client_id, sample_indices in user_groups.items():
        sample_indices = torch.tensor(list(sample_indices), dtype=torch.long, device=scores.device)
        client_influence_score = scores[sample_indices].sum().item()
        client_influences[client_id] = client_influence_score    
    return client_influences



def compute_influence_edb(args, delta_t_i, epoch):
    """Compute distances for each client using the method from Efficient Debugging."""
    client_influences = defaultdict(float)

    for ep in range(epoch // 2, epoch):
        for cid, delta in delta_t_i[ep].items():
            flat = torch.cat([tensor.view(-1) for tensor in delta.values()])
            norm = torch.norm(flat, p=2).item()
            client_influences[cid] += norm

    print(client_influences)
    return client_influences