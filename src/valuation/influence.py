import copy
from collections import defaultdict
import torch
import torch.nn.functional as F
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

    def test_loss(self, model, params, batch):
        return F.cross_entropy(model(batch[0]), batch[1]) 


def compute_influence(args, global_weights, train_dataset, test_dataset, user_groups):
    device = get_device()
    
    t_dataset = copy.deepcopy(train_dataset)
    
    train_loader = DataLoader(t_dataset.dataset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    # prepare the module
    model = initialize_model(args)
    model.load_state_dict(global_weights)
    model.to(device).float()
    model.eval()
    module = LiSSAInfluenceModule(model=model, objective=MyObjective(), train_loader=train_loader, test_loader=test_loader, device=device, damp=0.001, repeat=10, depth=5, scale=1.0)
   
    # load scores
    client_influences = defaultdict(float)
    for client_id, sample_indices in user_groups.items():
        print(f"Computing Influence For Client: {client_id}")
        test_indices = np.random.choice(len(test_dataset), 1024, replace=False).tolist()
        client_influences[client_id] = module.influences(list(sample_indices), test_indices).sum().item()

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