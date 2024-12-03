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


def compute_influence(args, global_weights, train_dataset, test_dataset, user_groups):
    """Calculate client-wise influence values."""
    device = get_device()
    model = initialize_model(args)
    model.load_state_dict(global_weights)
    model.to(device).float()
    model.eval()

    train_loader = DataLoader(train_dataset, batch_size=args.local_bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.local_bs, shuffle=False)

    print(f'Computing Influence Functions')

    config = ptif.get_default_config()
    config.device = device
    influences, harmful, helpful = ptif.calc_img_wise(config, model, train_loader, test_loader)

    client_influences = defaultdict(float)
    for id, indexes in user_groups.items():
        client_influences[id] = sum(influences[indexes])

    return client_influences