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



def graph_processed_log(log_file):
    # collect data from each setting
    acc_loss_before, acc_loss_after, bca_simple, bca_hessian = process_log(log_file)
    
    acc_before = np.array([round(float(item[0]), 2) for item in acc_loss_before])
    loss_before = np.array([round(float(item[1]), 2) for item in acc_loss_before])
    acc_after = np.array([round(float(item[0]), 2) for item in acc_loss_after])
    loss_after = np.array([round(float(item[1]), 2) for item in acc_loss_after])
    
    print("Accuracy Before:", acc_before)
    print("Accuracy After:", acc_after)
    print("Loss Before:", loss_before)
    print("Loss After:", loss_after)

    bad_client_percentages = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    x = range(1, len(acc_before) + 1)  # X-axis as setting indices

    # Generate plots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 6))

    # Accuracy Plot
    axes[0].plot(x, acc_before, marker='o', label='Initial Accuracy')
    axes[0].plot(x, acc_after, marker='s', label='Retrained Accuracy')
    axes[0].set_title("Accuracy Before and After Retraining")
    axes[0].set_xlabel("Bad Client Percentage")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_xticks(x, [f"{perc:.1f}" for perc in bad_client_percentages], rotation=0)
    axes[0].set_ylim(0, 100)  # Set fixed y-axis limits for accuracy
    axes[0].legend()
    axes[0].grid(True)

    # Loss Plot
    axes[1].plot(x, loss_before, marker='o', label='Initial Loss')
    axes[1].plot(x, loss_after, marker='s', label='Retrained Loss')
    axes[1].set_title("Loss Before and After Retraining")
    axes[1].set_xlabel("Setting")
    axes[1].set_ylabel("Loss")
    axes[1].set_xticks(x, [f"{perc:.1f}%" for perc in bad_client_percentages], rotation=0)
    min_loss = min(loss_before.min(), loss_after.min())
    max_loss = max(loss_before.max(), loss_after.max())
    axes[1].set_ylim(min_loss * 0.95, max_loss * 1.05)  # Add some padding
    axes[1].legend()
    axes[1].grid(True)

    # Bad Client Accuracy Detection Plot
    bar_width = 0.35
    indices = np.arange(len(x))
    axes[2].bar(indices, bca_simple, width=bar_width, label='Simple', align='center')
    axes[2].bar(indices + bar_width, bca_hessian, width=bar_width, label='Hessian', align='center')
    axes[2].set_title("Bad Client Accuracy Detection")
    axes[2].set_xlabel("Setting")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_xticks(indices + bar_width / 2, [f"{perc:.1f}%" for perc in bad_client_percentages], rotation=0)
    axes[2].legend()
    axes[2].grid(True)

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


graph_processed_log('retraining/cifar3.log')