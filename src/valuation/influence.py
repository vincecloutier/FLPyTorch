import torch
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

def compute_influence(args, model, train_dataset, test_dataset, user_groups):
    """Estimate Influence values for participants in a round using permutation sampling."""
    device = get_device()
    model.to(device)
    model.eval()

    influences = defaultdict(float)
    print(f'Computing Influence Functions for {args.num_users} clients')
    for id, indexes in user_groups.items():
        
        print(f'here')
        train_data_loader = DataLoader(ClientSplit(train_dataset, indexes), batch_size=args.local_bs, shuffle=False)
        print(f'here2')
        influence_model = NystroemSketchInfluence(model, F.cross_entropy, rank = 10, hessian_regularization=0.1)
        print(f'here3')
        influence_model = influence_model.fit(train_data_loader)
        print(f'here4')
        all_influences = influence_model.influences(*test_dataset, *train_dataset, mode="up")
        print(f'here5')
        influences[id] = sum(np.mean(all_influences.numpy(), axis=0))
        print(f'here6')

    return influences