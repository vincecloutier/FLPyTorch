import torch
from torch import nn, autocast
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler 
from utils import get_device


class ClientSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class."""
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.trainloader = DataLoader(ClientSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.device = get_device()
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def update_weights(self, model, global_round):
        # set mode to train model
        model.train()
        epoch_loss = []

        scaler = GradScaler()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)        
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.args.lr, epochs=self.args.local_ep, steps_per_epoch=len(self.trainloader))

        for iter in range(self.args.local_ep):
            batch_loss = []
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                with autocast(device_type='cuda', dtype=torch.float16):
                    output = model(images)
                    loss = self.criterion(output, labels)
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()

                sched.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

            print(f'| Global Round : {global_round+1} | Local Epoch : {iter+1} | Loss: {sum(batch_loss) / len(batch_loss):.6f}')

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    # this was used for pcc
    # def update_weights(self, model, global_round):
    #     # set mode to train model
    #     model.train()
    #     epoch_loss = []

    #     optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)
    #     sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.args.lr, epochs=self.args.local_ep, steps_per_epoch=len(self.trainloader))

    #     for iter in range(self.args.local_ep):
    #         batch_loss = []
    #         for images, labels in self.trainloader:
    #             images, labels = images.to(self.device), labels.to(self.device)

    #             model.zero_grad()
    #             log_probs = model(images)
    #             loss = self.criterion(log_probs, labels)
    #             loss.backward()

    #             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    #             optimizer.step()
    #             sched.step()

    #             batch_loss.append(loss.item())

    #         epoch_loss.append(sum(batch_loss)/len(batch_loss))

    #         print(f'| Global Round : {global_round+1} | Local Epoch : {iter+1} | Loss: {sum(batch_loss) / len(batch_loss):.6f}')

    #     return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


def test_inference(model, test_dataset):
    """Returns the test accuracy and loss on the global model trained on the entire dataset."""

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = get_device()

    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # forward pass
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total
    return accuracy, loss


def gradient(args, model, dataset):
    """Computes the gradient of the validation loss with respect to the model parameters."""

    model.eval()
    model.zero_grad()  # clear existing gradients

    device = get_device()

    criterion = nn.CrossEntropyLoss().to(device)
    data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    
    inputs, targets = next(iter(data_loader))
    inputs, targets = inputs.to(device), targets.to(device)

    # forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # backward pass
    grad_params = torch.autograd.grad(loss, model.parameters(), create_graph=False, retain_graph=False)

    # collect gradients
    gradient = {}
    for (name, param), grad in zip(model.named_parameters(), grad_params):
        if param.requires_grad:
            gradient[name] = grad.clone().detach()

    del inputs, targets, outputs, loss, grad_params
    torch.cuda.empty_cache()

    return gradient


# this one was used for pcc (as of now)
# def compute_hessian(model, dataset, v_list):
#     """Computes the Hessian-vector product Hv, where H is the Hessian of loss w.r.t. model parameters."""
#     model.eval()
#     model.zero_grad()

#     device = get_device()

#     criterion = nn.CrossEntropyLoss().to(device)
#     data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

#     inputs, targets = next(iter(data_loader))
#     inputs, targets = inputs.to(device), targets.to(device)

#     # forward pass
#     outputs = model(inputs)
#     loss = criterion(outputs, targets)

#     # first gradient
#     params = [p for p in model.parameters() if p.requires_grad]
#     grad_params = torch.autograd.grad(loss, params, create_graph=True)

#     # flatten grad_params and v_list
#     grad_params_flat = torch.cat([g.contiguous().view(-1) for g in grad_params])
#     v_flat = torch.cat([v.contiguous().view(-1) for v in v_list])

#     # compute the dot product grad_params_flat * v_flat
#     grad_dot_v = torch.dot(grad_params_flat, v_flat)
    
#     # second gradient
#     hvp = torch.autograd.grad(grad_dot_v, params)
    
#     del outputs, loss, grad_params, grad_params_flat, v_flat, grad_dot_v
#     torch.cuda.empty_cache()

#     # return as a dict
#     hv_dict = {}
#     for (name, param), hv in zip(model.named_parameters(), hvp):
#         if param.requires_grad:
#             hv_dict[name] = hv.clone().detach()
        
#     return hv_dict


def compute_hessian(model, dataset, v_list):
    """Computes the Hessian-vector product Hv by averaging over batches."""
    model.eval()
    device = get_device()
    criterion = nn.CrossEntropyLoss().to(device)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False)
    params = [p for p in model.parameters() if p.requires_grad]
    hvp_total = [torch.zeros_like(p) for p in params]
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        grad_params = torch.autograd.grad(loss, params, create_graph=True)
        grad_params_flat = torch.cat([g.contiguous().view(-1) for g in grad_params])
        v_flat = torch.cat([v.contiguous().view(-1) for v in v_list])
        grad_dot_v = torch.dot(grad_params_flat, v_flat)
        hvp = torch.autograd.grad(grad_dot_v, params)
        for i, hv in enumerate(hvp):
            hvp_total[i] += hv.detach()
        model.zero_grad()
    hvp_avg = [hv / len(data_loader) for hv in hvp_total]
    hv_dict = {name: hv.clone().detach() for (name, _), hv in zip(model.named_parameters(), hvp_avg) if _.requires_grad}

    del hvp_total, hvp_avg, hvp
    torch.cuda.empty_cache()

    return hv_dict
