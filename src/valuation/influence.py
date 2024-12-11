import copy
from collections import defaultdict
from typing import Tuple
import torch
import torch.nn.functional as F
from torch import nn
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.task import Task
from kronfluence.utils.common.factor_arguments import all_low_precision_factor_arguments
from kronfluence.utils.common.score_arguments import all_low_precision_score_arguments
from kronfluence.utils.dataset import DataLoaderKwargs
from utils import get_device, initialize_model

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
        # Copied from: https://github.com/MadryLab/trak/blob/main/trak/modelout_functions.py.
        inputs, labels = batch
        logits = model(inputs)

        bindex = torch.arange(logits.shape[0]).to(device=logits.device, non_blocking=False)
        logits_correct = logits[bindex, labels]

        cloned_logits = logits.clone()
        cloned_logits[bindex, labels] = torch.tensor(-torch.inf, device=logits.device, dtype=logits.dtype)

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return -margins.sum()


def compute_influence(args, global_weights, train_dataset, test_dataset, user_groups, noise_transform):
    device = get_device()

    # applying noise transform to train_dataset
    t_dataset = copy.deepcopy(train_dataset)
    t_dataset.data = [torch.tensor(data, dtype=torch.float32, device=device) for data in t_dataset.data]
    t_dataset.data = [noise_transform(d) for d in t_dataset.data]

    # prepare the model
    model = initialize_model(args)
    model.load_state_dict(global_weights)
    model.to(device).float()
    model.eval()

    # define task and prepare model
    task = ClassificationTask()
    model = prepare_model(model, task)
    analyzer = Analyzer(analysis_name="cifar10", model=model, task=task, profile=False)

    # configure parameters
    dataloader_kwargs = DataLoaderKwargs(num_workers=4)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    # compute influence factors
    factors_name = "ekfac"
    factor_args = FactorArguments(strategy=factors_name)
    # use half precision
    # factor_args = all_low_precision_factor_arguments(strategy=factors_name, dtype=torch.bfloat16)
    # factors_name += "_half"
    # fit all factors
    analyzer.fit_all_factors(
        factors_name=factors_name,
        factor_args=factor_args,
        dataset=t_dataset,
        per_device_batch_size=None,
        overwrite_output_dir=True,
    )

    # compute pairwise scores
    score_args = ScoreArguments()
    scores_name = factor_args.strategy
    # use half precision
    # score_args = all_low_precision_score_arguments(dtype=torch.bfloat16)
    # scores_name += "_half"
    # compute scores
    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        score_args=score_args,
        factors_name=factors_name,
        query_dataset=test_dataset,
        query_indices=list(range(1000)),
        train_dataset=t_dataset,
        per_device_query_batch_size=128,
        overwrite_output_dir=True
    )
    scores = analyzer.load_pairwise_scores(scores_name)["all_modules"]
    print(f"Scores shape: {scores.shape}")

    client_influences = defaultdict(float)
    
    # sum influence scores over all test samples for each training sample -> shape: [num_train_samples]
    influence_scores = scores.sum(dim=0)
    influence_scores = influence_scores / influence_scores.max()  # normalize
    
    # iterate over each client and sum the influence scores of their training samples
    for client_id, sample_indices in user_groups.items():
        # convert sample_indices to a tensor if they aren't already
        sample_indices = torch.tensor(list(sample_indices), dtype=torch.long, device=scores.device)
        # aggregate the influence scores for the client's training samples
        client_influence_score = influence_scores[sample_indices].sum().item()
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