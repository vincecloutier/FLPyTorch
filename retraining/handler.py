import re
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np
from utils import identify_bad_idxs, measure_accuracy

def process_log(file_path):
    # read the log file
    with open(file_path, 'r') as file:
        logs = file.read()

    # regex patterns
    approx_simple_pattern = r"Banzhaf Values Simple: defaultdict\(<class 'float'>, \{(.*?)\}\)"
    approx_hessian_pattern = r"Banzhaf Values Hessian: defaultdict\(<class 'float'>, \{(.*?)\}\)"
    before_pattern = r"Test Accuracy Before Retraining: ([\d\.]+), Test Loss Before Retraining: ([\d\.]+)"
    after_pattern = r"Test Accuracy After Retraining: ([\d\.]+), Test Loss After Retraining: ([\d\.]+)"
    bad_clients_pattern = r"Actual Bad Clients: \[([-\d\s,]+)\]"

    # extract values
    abv_simple = [eval(f"{{{match}}}") for match in re.findall(approx_simple_pattern, logs)]
    abv_hessian = [eval(f"{{{match}}}") for match in re.findall(approx_hessian_pattern, logs)]
    acc_loss_before = [eval(f"{{{match}}}") for match in re.findall(before_pattern, logs)]
    acc_loss_after = [eval(f"{{{match}}}") for match in re.findall(after_pattern, logs)]
    bad_clients = [[int(num) for num in re.findall(r'-?\d+', match)] for match in re.findall(bad_clients_pattern, logs) if match]

    bca_simple, bca_hessian = [], []
    for i in range(len(bad_clients)):
        bca_simple.append(measure_accuracy(bad_clients[i], identify_bad_idxs(abv_simple[i])))
        bca_hessian.append(measure_accuracy(bad_clients[i], identify_bad_idxs(abv_hessian[i])))

    return acc_loss_before, acc_loss_after, bca_simple, bca_hessian

process_log('retraining/cifar3.log')



# process_and_graph_logs('retraining/cifar1.log',  'retraining/cifar2.log', 'retraining/cifar3.log')
