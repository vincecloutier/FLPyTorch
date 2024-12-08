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
    before_pattern = r"Test Accuracy Before Retraining: ([\d\.]+), Test Loss Before Retraining: ([\d\.]+)"
    after_pattern = r"Test Accuracy After Retraining: ([\d\.]+), Test Loss After Retraining: ([\d\.]+)"

    # extract all values
    acc_loss_before = [tuple(eval(f"({match})")) for match in re.findall(before_pattern, logs)]
    acc_loss_after = [tuple(eval(f"({match})")) for match in re.findall(after_pattern, logs)]
    
    # 1 is the lowest quality data setting, 3 is the highest
    acc_loss_before_1 = [al for i, al in enumerate(acc_loss_before) if i in [0, 3, 6, 9]]
    acc_loss_after_1 = [al for i, al in enumerate(acc_loss_after) if i in [0, 3, 6, 9]]

    acc_loss_before_2 = [al for i, al in enumerate(acc_loss_before) if i in [1, 4, 7, 10]]
    acc_loss_after_2 = [al for i, al in enumerate(acc_loss_after) if i in [1, 4, 7, 10]]

    acc_loss_before_3 = [al for i, al in enumerate(acc_loss_before) if i in [2, 5, 8, 11]]
    acc_loss_after_3 = [al for i, al in enumerate(acc_loss_after) if i in [2, 5, 8, 11]]

    # convert to numpy arrays
    ab1 = np.array([round(float(item[0]), 2) for item in acc_loss_before_1])
    lb1 = np.array([round(float(item[1]), 2) for item in acc_loss_before_1])
    aa1 = np.array([round(float(item[0]), 2) for item in acc_loss_after_1])
    la1 = np.array([round(float(item[1]), 2) for item in acc_loss_after_1])

    ab2 = np.array([round(float(item[0]), 2) for item in acc_loss_before_2])
    lb2 = np.array([round(float(item[1]), 2) for item in acc_loss_before_2])
    aa2 = np.array([round(float(item[0]), 2) for item in acc_loss_after_2])
    la2 = np.array([round(float(item[1]), 2) for item in acc_loss_after_2])

    ab3 = np.array([round(float(item[0]), 2) for item in acc_loss_before_3])
    lb3 = np.array([round(float(item[1]), 2) for item in acc_loss_before_3])
    aa3 = np.array([round(float(item[0]), 2) for item in acc_loss_after_3])
    la3 = np.array([round(float(item[1]), 2) for item in acc_loss_after_3])

    return ab1, lb1, aa1, la1, ab2, lb2, aa2, la2, ab3, lb3, aa3, la3


def make_accuracy_plot(ax, x, acc_before, acc_after, x_label):
    ax.plot(x, acc_before, marker='o', label='Initial Accuracy')
    ax.plot(x, acc_after, marker='s', label='Retrained Accuracy')
    if ax == [0, 0]:
        ax.set_title("Accuracy Before and After Retraining")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x, [f"{n}" for n in range(2, 10, 2)], rotation=0)
    ax.set_ylim(0, 100)  # set fixed y-axis limits for accuracy
    ax.legend()
    ax.grid(True)

def make_loss_plot(ax, x, loss_before, loss_after, x_label, min_loss, max_loss):
    ax.plot(x, loss_before, marker='o', label='Initial Loss')
    ax.plot(x, loss_after, marker='s', label='Retrained Loss')
    if ax == 0:
        ax.set_title("Loss Before and After Retraining")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Loss")
    ax.set_xticks(x, [f"{n}" for n in range(2, 10, 2)], rotation=0)
    ax.set_ylim(min_loss * 0.95, max_loss * 1.05)  # add some padding
    ax.legend()
    ax.grid(True)

def graph_processed_log(log_file):
    # collect data from each setting
    if "1" in log_file:
        x_label = "Number of Clients with Non-IID Data"
    elif "2" in log_file:
        x_label = "Number of Clients with Mislabeled Data"
    elif "3" in log_file:
        x_label = "Number of Clients with Noisy Data"

    ab1, lb1, aa1, la1, ab2, lb2, aa2, la2, ab3, lb3, aa3, la3 = process_log(log_file)

    # x-axis as setting indices
    x = range(1, len(ab1) + 1)

    # generate plots
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))

    # use the loss from the lowest quality data setting to set the y-axis limits for all plots
    min_loss = min(lb1.min(), lb2.min(), lb3.min())
    max_loss = max(lb1.max(), lb2.max(), lb3.max())

    # accuracy and loss plots for lowest quality data setting
    make_accuracy_plot(axes[0, 0], x, ab1, aa1, x_label)
    make_loss_plot(axes[0, 1], x, lb1, la1, x_label, min_loss, max_loss)

    # accuracy and loss plots for middle quality data setting
    make_accuracy_plot(axes[1, 0], x, ab2, aa2, x_label)
    make_loss_plot(axes[1, 1], x, lb2, la2, x_label, min_loss, max_loss)

    # accuracy and loss plots for highest quality data setting
    make_accuracy_plot(axes[2, 0], x, ab3, aa3, x_label)
    make_loss_plot(axes[2, 1], x, lb3, la3, x_label, min_loss, max_loss)

    plt.tight_layout()
    plt.savefig(f"retraining/graphs/{log_file.split('/')[-1].split('.')[0]}.png")


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

graph_processed_log('retraining/resnet2.log')