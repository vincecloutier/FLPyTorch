import re
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np
from collections import defaultdict
import torch
def process_log(file_path):
    # read the log file
    with open(file_path, 'r') as file:
        logs = file.read()

    # regex patterns
    approx_simple_pattern = r"Banzhaf Values Simple: defaultdict\(<class 'float'>, \{(.*?)\}\)"
    approx_hessian_pattern = r"Banzhaf Values Hessian: defaultdict\(<class 'float'>, \{(.*?)\}\)"
    actual_bad_clients_pattern = r"Actual Bad Clients: \[(.*?)\]"
    before_pattern = r"Test Accuracy Before Retraining: ([\d\.]+), Test Loss Before Retraining: ([\d\.]+)"
    after_pattern = r"Test Accuracy After Retraining: ([\d\.]+), Test Loss After Retraining: ([\d\.]+)"
    bad_client_accuracy_simple_pattern = r"Bad Client Accuracy Simple: (.*?)\n"
    bad_client_accuracy_hessian_pattern = r"Bad Client Accuracy Hessian: (.*?)\n"

    # extract values
    approx_simple_values = [eval(f"{{{match}}}") for match in re.findall(approx_simple_pattern, logs)]
    approx_hessian_values = [eval(f"{{{match}}}") for match in re.findall(approx_hessian_pattern, logs)]
    before_values = [eval(f"{{{match}}}") for match in re.findall(before_pattern, logs)]
    after_values = [eval(f"{{{match}}}") for match in re.findall(after_pattern, logs)]
    bad_client_accuracy_simple_values = [eval(f"{{{match}}}") for match in re.findall(bad_client_accuracy_simple_pattern, logs)]
    bad_client_accuracy_hessian_values = [eval(f"{{{match}}}") for match in re.findall(bad_client_accuracy_hessian_pattern, logs)]
    actual_bad_clients_values = [eval(f"{{{match}}}") for match in re.findall(actual_bad_clients_pattern, logs)]

    # print(approx_simple_values)
    # print(approx_hessian_values)
    # print(before_values)
    # print(after_values)
    # print(bad_client_accuracy_simple_values)
    # print(bad_client_accuracy_hessian_values)
    # print(actual_bad_clients_values)

    new_bad_client_accuracy_simple_values = []
    new_bad_client_accuracy_hessian_values = []
    for i in range(len(bad_client_accuracy_simple_values)):
        bad_clients = actual_bad_clients_values[i]
        abv_simple = approx_simple_values[i]
        abv_hessian = approx_hessian_values[i]
        new_bad_client_accuracy_simple_values.append(measure_accuracy(bad_clients, identify_bad_idxs(abv_simple)))
        new_bad_client_accuracy_hessian_values.append(measure_accuracy(bad_clients, identify_bad_idxs(abv_hessian)))

    print(f"Simple: {bad_client_accuracy_simple_values} / {new_bad_client_accuracy_simple_values}")
    print(f"Hessian: {bad_client_accuracy_hessian_values} / {new_bad_client_accuracy_hessian_values}")



def identify_bad_idxs(approx_banzhaf_values: dict, threshold: float = 3) -> list[int]:
    if not approx_banzhaf_values:
        return []
    banzhaf_tensor = torch.tensor(list(approx_banzhaf_values.values()))
    median_banzhaf = torch.median(banzhaf_tensor)    
    bad_idxs = [key for key, banzhaf in approx_banzhaf_values.items() if (banzhaf < median_banzhaf / threshold)]
    # bad_idxs = [key for key, banzhaf in approx_banzhaf_values.items() if banzhaf < 0]
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


process_log('retraining/cifar1.log')



# process_and_graph_logs('retraining/cifar1.log',  'retraining/cifar2.log', 'retraining/cifar3.log')
