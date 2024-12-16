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
from torch_influence import BaseObjective
from torch_influence import LiSSAInfluenceModule
from torch.utils.data import DataLoader
import numpy as np

class MyObjective(BaseObjective):
    def train_outputs(self, model, batch):
        return model(batch[0])

    def train_loss_on_outputs(self, outputs, batch):
        return F.cross_entropy(outputs, batch[1]) 

    def train_regularization(self, params):
        return 0.01 * torch.square(params.norm())

    # training loss by default taken to be 
    # train_loss_on_outputs + train_regularization

    def test_loss(self, model, params, batch):
        return F.cross_entropy(model(batch[0]), batch[1])  # no regularization in test loss


def compute_influence(args, global_weights, train_dataset, test_dataset, user_groups, noise_transform = None):
    device = get_device()
    
    t_dataset = copy.deepcopy(train_dataset)
    if noise_transform is not None:
        # applying noise transform to train_dataset
        noise_transform.to('cpu')
        t_dataset.data = [noise_transform(torch.tensor(data, dtype=torch.float32)) for data in t_dataset.data]
        noise_transform.to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    # prepare the model
    model = initialize_model(args)
    model.load_state_dict(global_weights)
    model.to(device).float()
    model.eval()

    module = LiSSAInfluenceModule(
        model=model,
        objective=MyObjective(),  
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        damp=0.001,
        repeat=10,
        depth=5,
        scale = 1.0,   
    )

    # load scores
    client_influences = defaultdict(float)
    for client_id, sample_indices in user_groups.items():
        test_indices = np.random.choice(len(test_dataset), 1000, replace=False).tolist()
        print(module.influences(sample_indices, test_indices))
        break

    return client_influences



BATCH_TYPE = Tuple[torch.Tensor, torch.Tensor]


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


# def compute_influence(args, global_weights, train_dataset, user_groups, noise_transform = None):
#     device = get_device()
    
#     t_dataset = copy.deepcopy(train_dataset)
#     if noise_transform is not None:
#         # applying noise transform to train_dataset
#         noise_transform.to('cpu')
#         t_dataset.data = [noise_transform(torch.tensor(data, dtype=torch.float32)) for data in t_dataset.data]
#         noise_transform.to(device)
    
#     # prepare the model
#     model = initialize_model(args)
#     model.load_state_dict(global_weights)
#     model.to(device).float()
#     model.eval()

#     # define task and prepare model
#     task = ClassificationTask()
#     model = prepare_model(model, task)
#     analyzer = Analyzer(analysis_name="analysis", model=model, task=task, profile=False)

#     # configure parameters
#     dataloader_kwargs = DataLoaderKwargs(num_workers=4)
#     analyzer.set_dataloader_kwargs(dataloader_kwargs)

#     # compute influence factors
#     factor_args = FactorArguments(
#         strategy=args.strategy,
#         use_empirical_fisher=True,
#         amp_dtype=torch.bfloat16,
#         amp_scale=2.0**16,
#         # precision settings
#         eigendecomposition_dtype=torch.float32,
#         activation_covariance_dtype=torch.bfloat16,
#         gradient_covariance_dtype=torch.bfloat16,
#         per_sample_gradient_dtype=torch.bfloat16,
#         lambda_dtype=torch.bfloat16,
#     )
#     analyzer.fit_all_factors(
#         factors_name=args.strategy,
#         dataset=t_dataset,
#         per_device_batch_size=None,
#         factor_args=factor_args,
#         overwrite_output_dir=True,
#     )

#     # compute influence scores
#     score_args = ScoreArguments(
#         damping_factor=1e-1,
#         amp_dtype=torch.bfloat16,
#         use_measurement_for_self_influence=True,
#         # precision settings
#         query_gradient_svd_dtype=torch.bfloat16,
#         per_sample_gradient_dtype=torch.bfloat16,
#         precondition_dtype=torch.bfloat16,
#         score_dtype=torch.bfloat16
#     )

#     analyzer.compute_self_scores(
#         scores_name=args.strategy,
#         factors_name=args.strategy,
#         train_dataset=t_dataset,
#         score_args=score_args,
#         overwrite_output_dir=True,
#     )

#     # load scores
#     scores = analyzer.load_self_scores(args.strategy)["all_modules"]
#     print(f"Scores shape: {scores.shape}")
#     client_influences = defaultdict(float)
#     for client_id, sample_indices in user_groups.items():
#         sample_indices = torch.tensor(list(sample_indices), dtype=torch.long, device=scores.device)
#         client_influence_score = scores[sample_indices].sum().item()
#         client_influences[client_id] = client_influence_score    
#     return client_influences



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