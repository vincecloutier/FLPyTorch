import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
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
        return image, torch.tensor(label, dtype=torch.long)


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

        # set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            if self.args.verbose:
                print(f'| Global Round : {global_round+1} | Local Epoch : {iter+1} | Loss: {sum(batch_loss) / len(batch_loss):.6f}')

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


def test_inference(model, test_dataset):
    """Returns the test accuracy and loss on the global model trained on the entire dataset."""

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = get_device()

    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            # inference
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item() * len(labels)

            # prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

    accuracy = correct / total
    loss = loss / total
    return accuracy, loss


def test_gradient(args, model, dataset):
    """Computes the gradient of the validation loss with respect to the model parameters."""

    model.train()
    model.zero_grad()  # clear existing gradients

    device = get_device()

    criterion = nn.CrossEntropyLoss().to(device)
    data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False) # top insight to use full dataset
    
    inputs, targets = next(iter(data_loader))
    inputs, targets = inputs.to(device), targets.to(device)

    # forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # backward pass
    if args.hessian == 1:
        grad_params = torch.autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
    else:
        grad_params = torch.autograd.grad(loss, model.parameters(), create_graph=False, retain_graph=False)

    # collect gradients
    gradient = {}
    for (name, param), grad in zip(model.named_parameters(), grad_params):
        if param.requires_grad:
            gradient[name] = grad.clone().detach()

    return gradient


def compute_hessian(model, dataset, v_list):
    """Computes the Hessian-vector product Hv, where H is the Hessian of loss w.r.t. model parameters."""
    device = get_device()
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    inputs, targets = next(iter(data_loader))
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model(inputs)
    validation_loss = criterion(outputs, targets)

    # first gradient
    params = [p for p in model.parameters() if p.requires_grad]
    grad_params = torch.autograd.grad(validation_loss, params, create_graph=True, retain_graph=True)
    
    # flatten grad_params and v_list
    grad_params_flat = torch.cat([g.contiguous().view(-1) for g in grad_params])
    v_flat = torch.cat([v.contiguous().view(-1) for v in v_list])
    
    # compute the dot product grad_params_flat * v_flat
    grad_dot_v = torch.dot(grad_params_flat, v_flat)
    
    # second gradient
    hvp = torch.autograd.grad(grad_dot_v, params, retain_graph=True)
    
    del outputs, validation_loss, grad_params, grad_params_flat, v_flat, grad_dot_v
    torch.cuda.empty_cache()

    # return as a dict
    hv_dict = {}
    for (name, param), hv in zip(model.named_parameters(), hvp):
        if param.requires_grad:
            hv_dict[name] = hv.clone().detach()
    return hv_dict



def conjugate_gradient(model, dataset, b, num_iterations=10, tol=1e-10):
    """Solves Hx = b using the Conjugate Gradient method, where H is the Hessian."""
    device = get_device()
    model.eval()
    
    # initialize x to zero
    x = torch.zeros_like(b).to(device)
    r = b.clone().detach()  # initial residual r0 = b - Hx0 = b
    p = r.clone().detach()  # initial search direction p0 = r0
    rs_old = torch.dot(r, r)
    
    for i in range(num_iterations):
        # split x and p into list of tensors matching model parameters
        x_list = torch.split(x, [p.numel() for p in model.parameters() if p.requires_grad])
        p_list = torch.split(p, [p.numel() for p in model.parameters() if p.requires_grad])
        
        # reshape the split tensors to match parameter shapes
        p_reshaped = []
        idx = 0
        for param in model.parameters():
            if param.requires_grad:
                numel = param.numel()
                p_reshaped.append(p[idx:idx+numel].view_as(param))
                idx += numel
        v_list = p_reshaped
        
        # compute Hv using the Hessian-vector product
        Hv_dict = compute_hessian(model, dataset, v_list)
        
        # flatten Hv_dict into a single vector
        Hv = torch.cat([Hv_dict[name].contiguous().view(-1) for name, _ in model.named_parameters() if _.requires_grad])
        
        alpha = rs_old / torch.dot(p, Hv)
        x += alpha * p
        r -= alpha * Hv
        rs_new = torch.dot(r, r)
        
        if torch.sqrt(rs_new) < tol:
            print(f'Conjugate Gradient converged in {i+1} iterations.')
            break
        
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    
    return x