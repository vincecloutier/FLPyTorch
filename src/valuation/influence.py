from tqdm import tqdm
from collections import defaultdict
from update import LocalUpdate, gradient, conjugate_gradient, ClientSplit
import multiprocessing
from functools import partial
from utils import get_device, initialize_model, average_weights
import numpy as np
from pydvl.influence.torch import EkfacInfluence, NystroemSketchInfluence
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_influence_functions as ptif
import os
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


# def compute_influence(args, global_weights, train_dataset, test_dataset, user_groups):
#     """Calculate client-wise influence values."""
#     device = get_device()
#     model = initialize_model(args)
#     model.load_state_dict(global_weights)
#     model.to(device).float()
#     model.eval()

#     train_loader = DataLoader(train_dataset, batch_size=args.local_bs, shuffle=False)
#     test_loader = DataLoader(test_dataset, batch_size=args.local_bs, shuffle=False)

#     print(f'Computing Influence Functions')

#     os.makedirs('outdir', exist_ok=True)

#     config = ptif.get_default_config()
#     # config['gpu'] = 0
#     config['recursion_depth'] = 1000
#     config['r'] = 5
#     influences, _, _ = ptif.calc_img_wise(config, model, train_loader, test_loader)

#     print("Sample influence entry:", next(iter(influences.values())))
#     print(f"Shape of influences: {np.array(next(iter(influences.values()))).shape}")

#     # convert user_groups to numpy arrays for faster indexing
#     user_groups_np = {cid: np.array(idxs, dtype=np.int32) for cid, idxs in user_groups.items()}
#     client_influences = defaultdict(float)
#     client_ids = list(user_groups_np.keys())
#     client_indices = list(user_groups_np.values())

#     for test_id, test_info in tqdm(influences.items(), total=len(influences), desc="Aggregating Influences"):
#         influence_scores = test_info.get('influence', [])
#         if not influence_scores:
#             continue  # skip if no influence scores

#         influence_array = np.array(influence_scores, dtype=np.float32)
        
#         for cid, indices in zip(client_ids, client_indices):
#             valid_indices = indices[indices < len(influence_array)]
#             client_influences[cid] += influence_array[valid_indices].sum()

#     return client_influences


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


def compute_influence(args, global_weights, train_dataset, test_dataset):
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
    factor_args = all_low_precision_factor_arguments(strategy=factors_name, dtype=torch.bfloat16)
    factors_name += "_half"
    # fit all factors
    analyzer.fit_all_factors(
        factors_name=factors_name,
        factor_args=factor_args,
        dataset=train_dataset,
        per_device_batch_size=None,
        overwrite_output_dir=False,
    )

    # compute pairwise scores
    score_args = ScoreArguments()
    scores_name = factor_args.strategy
    # use half precision (optional)
    score_args = all_low_precision_score_arguments(dtype=torch.bfloat16)
    scores_name += "_half"
    # compute scores
    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        score_args=score_args,
        factors_name=factors_name,
        query_dataset=test_dataset,
        query_indices=list(range(2000)),
        train_dataset=train_dataset,
        per_device_query_batch_size=1000,
        overwrite_output_dir=False,
    )
    scores = analyzer.load_pairwise_scores(scores_name)["all_modules"]
    print(f"Scores shape: {scores.shape}")

    