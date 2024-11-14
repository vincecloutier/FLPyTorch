import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from utils import get_device


class DatasetSplit(Dataset):
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
        self.trainloader = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.device = get_device()
        self.criterion = nn.NLLLoss().to(self.device)

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

    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss

def test_gradient(model, test_dataset):
    """Returns the average gradient of the loss with respect to the model parameters on the test dataset."""
    
    model.eval()
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()

    device = get_device()

    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    total_batches = len(testloader)
    
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels) / total_batches

        # backward pass to compute gradients
        loss.backward()

    # collect average gradients
    gradients = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            gradients[name] = param.grad.clone().detach() if param.grad is not None else None

    for key in gradients:
        gradients[key] = gradients[key].detach().to(device)

    return gradients