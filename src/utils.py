import copy
import torch
import logging
from models import CNNFashion, CNNCifar, ResNet9, MobileNetV2


def average_weights(w):
    """Returns the average of the weights."""
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def initialize_model(args):
    model_dict = {
        'fmnist': CNNFashion,
        'cifar': CNNCifar,
        'resnet': ResNet9,
        'mobilenet': MobileNetV2
    }
    if args.dataset in model_dict:
        return model_dict[args.dataset](args=args)
    else:
        exit('Error: unrecognized dataset')


def setup_logger(strategy_name: str) -> logging.Logger:
    """Set up a logger for the given strategy."""
    logger = logging.getLogger(strategy_name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f'{strategy_name}_metrics.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(fh)
    return logger


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def identify_bad_idxs(approx_banzhaf_values: dict, threshold: float = 2) -> list[int]:
    if not approx_banzhaf_values:
        return []
    banzhaf_tensor = torch.tensor(list(approx_banzhaf_values.values()))
    median_banzhaf = torch.median(banzhaf_tensor)    
    bad_idxs = [key for key, banzhaf in approx_banzhaf_values.items() if banzhaf > median_banzhaf / threshold]
    return bad_idxs


def measure_accuracy(targets, predictions):
    if targets is None or predictions is None:
        return 0.0
    if len(targets) == 0 and len(predictions) == 0:
        return 1.0
    targets, predictions = set(targets), set(predictions)
    TP = len(predictions & targets)
    FP = len(predictions - targets)
    FN = len(targets - predictions)
    universe = targets | predictions
    TN = len(universe - (targets | predictions))
    return (TP + TN) / (TP + TN + FP + FN)