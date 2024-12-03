from collections import defaultdict
from utils import get_device, initialize_model
import torch.nn.functional as F
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


def compute_influence(args, global_weights, train_dataset, test_dataset, user_groups):
    # prepare the model
    device = get_device()
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
    # use half precision (optional)
    # factor_args = all_low_precision_factor_arguments(strategy=factors_name, dtype=torch.bfloat16)
    # factors_name += "_half"
    # fit all factors
    analyzer.fit_all_factors(
        factors_name=factors_name,
        factor_args=factor_args,
        dataset=train_dataset,
        per_device_batch_size=None,
        overwrite_output_dir=True,
    )

    # compute pairwise scores
    score_args = ScoreArguments()
    scores_name = factor_args.strategy
    # use half precision (optional)
    # score_args = all_low_precision_score_arguments(dtype=torch.bfloat16)
    # scores_name += "_half"
    # compute scores
    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        score_args=score_args,
        factors_name=factors_name,
        query_dataset=test_dataset,
        query_indices=list(range(2000)),
        train_dataset=train_dataset,
        per_device_query_batch_size=1000,
        overwrite_output_dir=True,
    )
    scores = analyzer.load_pairwise_scores(scores_name)["all_modules"]
    print(f"Scores shape: {scores.shape}")

    client_influence = defaultdict(float)
    
    # sum influence scores over all test samples for each training sample -> shape: [num_train_samples]
    influence_scores = scores.sum(dim=0)
    influence_scores = influence_scores / influence_scores.max()  # normalize
    
    # iterate over each client and sum the influence scores of their training samples
    for client_id, sample_indices in user_groups.items():
        # convert sample_indices to a tensor if they aren't already
        sample_indices = torch.tensor(list(sample_indices), dtype=torch.long, device=scores.device)
        # aggregate the influence scores for the client's training samples
        client_influence_score = influence_scores[sample_indices].sum().item()
        client_influence[client_id] = client_influence_score
    
    return client_influence