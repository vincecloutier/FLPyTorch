import re
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np

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
    acc_loss_before = [tuple(eval(f"({match})")) for match in re.findall(before_pattern, logs)]
    acc_loss_after = [tuple(eval(f"({match})")) for match in re.findall(after_pattern, logs)]
    bad_clients = [[int(num) for num in re.findall(r'-?\d+', match)] for match in re.findall(bad_clients_pattern, logs) if match]
    print(bad_clients)
    print(abv_simple)
    print(acc_loss_before)
    bca_simple, bca_hessian = [], []
    for i in range(len(bad_clients)):
        bca_simple.append(measure_accuracy(bad_clients[i], identify_bad_idxs(abv_simple[i])))
        bca_hessian.append(measure_accuracy(bad_clients[i], identify_bad_idxs(abv_hessian[i])))

    return acc_loss_before, acc_loss_after, bca_simple, bca_hessian



def process_and_graph_logs(log_files):
    # collect data from each setting
    for log_file in log_files:
        acc_loss_before, acc_loss_after, bca_simple, bca_hessian = process_log(log_file)
        acc_before = [float(item[0]) for item in acc_loss_before]
        loss_before = [float(item[1]) for item in acc_loss_before]
        acc_after = [float(item[0]) for item in acc_loss_after]
        loss_after = [float(item[1]) for item in acc_loss_after]
        print(acc_before)
        print(acc_after)
        print(loss_before)
        print(loss_after)

        # generate plots
        plt.figure(figsize=(15, 5))

        # accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(['Initial Accuracy', 'Retrained Accuracy'], [acc_before, acc_after])
        plt.title(f"Accuracy Before and After Retraining")
        plt.xlabel("Bad Client Percentage")
        plt.ylabel("Accuracy")   

        # loss plot
        plt.subplot(1, 2, 2)
        plt.plot(['Initial Loss', 'Retrained Loss'], [loss_before, loss_after])
        # for i, v in enumerate([loss_before, loss_after]):
        #     plt
        plt.title(f"Loss Before and After Retraining")
        plt.xlabel("Bad Client Percentage")
        plt.ylabel("Loss")

        # bad client accuracy plot
        plt.subplot(1, 2, 2)
        plt.bar(['Simple', 'Hessian'], [bca_simple, bca_hessian])
        plt.title(f"Bad Client Accuracy Detection")
        plt.xlabel("Bad Client Percentage")
        plt.ylabel("Accuracy")

        plt.tight_layout()
        plt.show()


# TEMPORARY UTILS
def identify_bad_idxs(approx_banzhaf_values: dict, threshold: float = 2) -> list[int]:
    if not approx_banzhaf_values:
        return []

    # add all negative values to the list
    bad_idxs = [key for key, banzhaf in approx_banzhaf_values.items() if banzhaf < 0]

    # add all clients with banzhaf values less than the mean divided by the threshold to the list
    avg_banzhaf = np.mean(list(approx_banzhaf_values.values()))
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


process_and_graph_logs(['retraining/cifar1.log',  'retraining/cifar2.log', 'retraining/cifar3.log'])