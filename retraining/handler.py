import re
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np
from collections import defaultdict
import torch
from sklearn.preprocessing import MinMaxScaler
from mpmath import mp

def process_log(file_path):
    # read the log file
    with open(file_path, 'r') as file:
        logs = file.read()

    # regex patterns
    approx_simple_pattern = r"Banzhaf Values Simple: defaultdict\(<class 'float'>, \{(.*?)\}\)"
    approx_hessian_pattern = r"Banzhaf Values Hessian: defaultdict\(<class 'float'>, \{(.*?)\}\)"
    bad_clients_pattern = r"Actual Bad Clients: \[([-\d\s,]+)\]"
    before_pattern = r"Test Accuracy Before Retraining: ([\d\.]+), Test Loss Before Retraining: ([\d\.]+)"
    after_pattern = r"Test Accuracy After Retraining: ([\d\.]+), Test Loss After Retraining: ([\d\.]+)"
    bad_client_accuracy_simple_pattern = r"Bad Client Accuracy Simple: (.*?)\n"
    bad_client_accuracy_hessian_pattern = r"Bad Client Accuracy Hessian: (.*?)\n"

    # extract values
    approx_simple_values = [eval(f"{{{match}}}") for match in re.findall(approx_simple_pattern, logs)]
    approx_hessian_values = [eval(f"{{{match}}}") for match in re.findall(approx_hessian_pattern, logs)]
    before_values = [eval(f"{{{match}}}") for match in re.findall(before_pattern, logs)]
    after_values = [eval(f"{{{match}}}") for match in re.findall(after_pattern, logs)]
    bad_client_accuracy_simple_values = [float(match) for match in re.findall(bad_client_accuracy_simple_pattern, logs)]
    bad_client_accuracy_hessian_values = [float(match) for match in re.findall(bad_client_accuracy_hessian_pattern, logs)]
    all_bad_clients = [[int(num) for num in re.findall(r'-?\d+', match)] for match in re.findall(bad_clients_pattern, logs) if match]

    # print(approx_simple_values)
    # print(approx_hessian_values)
    # # print(before_values)
    # # print(after_values)
    # print(bad_client_accuracy_simple_values)
    # # print(bad_client_accuracy_hessian_values)
    # print(all_bad_clients)

    new_bad_client_accuracy_simple_values = []
    new_bad_client_accuracy_hessian_values = []
    for i in range(len(all_bad_clients)):
        bca_s = measure_accuracy(all_bad_clients[i], identify_bad_idxs(approx_simple_values[i]))
        bca_h = measure_accuracy(all_bad_clients[i], identify_bad_idxs(approx_hessian_values[i]))
        new_bad_client_accuracy_simple_values.append(bca_s)
        new_bad_client_accuracy_hessian_values.append(bca_h)
     
    print(f"Simple: {bad_client_accuracy_simple_values} / {new_bad_client_accuracy_simple_values}")
    print(f"Hessian: {bad_client_accuracy_hessian_values} / {new_bad_client_accuracy_hessian_values}")


def identify_bad_idxs(approx_banzhaf_values: dict, threshold: float = 2) -> list[int]:
    if not approx_banzhaf_values:
        return []
    # banzhaf_tensor = torch.tensor(list(approx_banzhaf_values.values()))
    # median_banzhaf = np.median(list(approx_banzhaf_values.values()))    
    # add all negative values to the list
    bad_idxs = [key for key, banzhaf in approx_banzhaf_values.items() if banzhaf < 0]

    # also if the mean is less than the threshold, add all values less than the mean to the list
    # approx_banzhaf_values = {k: v for k, v in approx_banzhaf_values.items() if k not in bad_idxs}
    avg_banzhaf = np.mean(list(approx_banzhaf_values.values()))
    med_banzhaf = np.median(list(approx_banzhaf_values.values()))
    bad_idxs.extend([key for key, banzhaf in approx_banzhaf_values.items() if banzhaf < avg_banzhaf / threshold])
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


process_log('retraining/cifar2.log')



# process_and_graph_logs('retraining/cifar1.log',  'retraining/cifar2.log', 'retraining/cifar3.log')
