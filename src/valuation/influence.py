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

    os.makedirs('outdir', exist_ok=True)

    config = ptif.get_default_config()
    config['gpu'] = 0
    config['recursion_depth'] = 1000
    config['r'] = 5
    influences, _, _ = ptif.calc_img_wise(config, model, train_loader, test_loader)

    print("Sample influence entry:", next(iter(influences.values())))
    print(f"Shape of influences: {np.array(next(iter(influences.values()))).shape}")

    # convert user_groups to numpy arrays for faster indexing
    user_groups_np = {cid: np.array(idxs, dtype=np.int32) for cid, idxs in user_groups.items()}
    client_influences = defaultdict(float)
    client_ids = list(user_groups_np.keys())
    client_indices = list(user_groups_np.values())

    for test_id, test_info in tqdm(influences.items(), total=len(influences), desc="Aggregating Influences"):
        influence_scores = test_info.get('influence', [])
        if not influence_scores:
            continue  # skip if no influence scores

        influence_array = np.array(influence_scores, dtype=np.float32)
        
        for cid, indices in zip(client_ids, client_indices):
            valid_indices = indices[indices < len(influence_array)]
            client_influences[cid] += influence_array[valid_indices].sum()

    return client_influences