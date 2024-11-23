import torch
from torch import nn
from torch.utils.data import DataLoader


def inference(args, model, test_dataset, device):
    """Returns the test accuracy and loss on the global model trained on the entire dataset."""

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        # forward pass
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


def gradient(args, model, test_dataset, device):
    """Computes the gradient of the validation loss with respect to the model parameters."""

    model.eval()
    model.zero_grad()  # clear existing gradients

    criterion = nn.CrossEntropyLoss().to(device)
    data_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=args.num_workers) # top insight to use full dataset
    
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


def hessian(args, model, test_dataset, v_list, device):
    """Computes the Hessian-vector product Hv, where H is the Hessian of loss w.r.t. model parameters."""
    
    model.eval()
    model.zero_grad()
    
    criterion = nn.CrossEntropyLoss().to(device)
    data_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    inputs, targets = next(iter(data_loader))
    inputs, targets = inputs.to(device), targets.to(device)
    
    # forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # first gradient
    params = [p for p in model.parameters() if p.requires_grad]
    grad_params = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    
    # flatten grad_params and v_list
    grad_params_flat = torch.cat([g.contiguous().view(-1) for g in grad_params])
    v_flat = torch.cat([v.contiguous().view(-1) for v in v_list])
    
    # compute the dot product grad_params_flat * v_flat
    grad_dot_v = torch.dot(grad_params_flat, v_flat)
    
    # second gradient
    hvp = torch.autograd.grad(grad_dot_v, params, retain_graph=True)
    
    # return as a dict
    hv_dict = {}
    for (name, param), hv in zip(model.named_parameters(), hvp):
        if param.requires_grad:
            hv_dict[name] = hv.clone().detach()
    return hv_dict