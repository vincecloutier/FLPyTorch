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
    before_pattern = r"Test Accuracy Before Retraining: ([\d\.]+), Test Loss Before Retraining: ([\d\.]+), in ([\d\.]+)s"
    after_pattern = r"Test Accuracy After Retraining: ([\d\.]+), Test Loss After Retraining: ([\d\.]+), in ([\d\.]+)s"

    # extract all values 
    acc_loss_before = [tuple(map(float, match)) for match in re.findall(before_pattern, logs)]
    acc_loss_after = [tuple(map(float, match)) for match in re.findall(after_pattern, logs)]

    # organize data into three quality settings
    def split_data(data, modulo):
        return [al for i, al in enumerate(data) if i % 3 == modulo]

    acc_loss_before_1 = split_data(acc_loss_before, 0)
    acc_loss_after_1 = split_data(acc_loss_after, 0)

    acc_loss_before_2 = split_data(acc_loss_before, 1)
    acc_loss_after_2 = split_data(acc_loss_after, 1)

    acc_loss_before_3 = split_data(acc_loss_before, 2)
    acc_loss_after_3 = split_data(acc_loss_after, 2)

    # convert to numpy arrays with rounding
    def to_rounded_array(data, index):
        return np.round(np.array([item[index] for item in data]), 2)

    ab1 = to_rounded_array(acc_loss_before_1, 0)
    lb1 = to_rounded_array(acc_loss_before_1, 1)
    aa1 = to_rounded_array(acc_loss_after_1, 0)
    la1 = to_rounded_array(acc_loss_after_1, 1)

    ab2 = to_rounded_array(acc_loss_before_2, 0)
    lb2 = to_rounded_array(acc_loss_before_2, 1)
    aa2 = to_rounded_array(acc_loss_after_2, 0)
    la2 = to_rounded_array(acc_loss_after_2, 1)

    ab3 = to_rounded_array(acc_loss_before_3, 0)
    lb3 = to_rounded_array(acc_loss_before_3, 1)
    aa3 = to_rounded_array(acc_loss_after_3, 0)
    la3 = to_rounded_array(acc_loss_after_3, 1)

    return ab1, lb1, aa1, la1, ab2, lb2, aa2, la2, ab3, lb3, aa3, la3


def make_accuracy_plot(ax, x, acc_before, acc_after, x_label):
    ax.plot(x, acc_before, marker='o', linestyle='-', color='tab:orange', label='Initial Accuracy')
    ax.plot(x, acc_after, marker='s', linestyle='--', color='tab:blue', label='Retrained Accuracy')
    ax.set_title("Accuracy Before and After Retraining", fontsize=14, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}" for n in x], fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)


def make_loss_plot(ax, x, loss_before, loss_after, x_label, min_loss, max_loss):
    ax.plot(x, loss_before, marker='o', linestyle='-', color='tab:orange', label='Initial Loss')
    ax.plot(x, loss_after, marker='s', linestyle='--', color='tab:blue', label='Retrained Loss')
    ax.set_title("Loss Before and After Retraining", fontsize=14, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}" for n in x], fontsize=10)
    ax.set_ylim(min_loss * 0.95, max_loss * 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)


def graph_processed_log(log_file):
    # collect data from each setting
    if "1" in log_file:
        x_label_1 = "Number of Clients with Non-IID Data (4 Categories)"
        x_label_2 = "Number of Clients with Non-IID Data (6 Categories)"
        x_label_3 = "Number of Clients with Non-IID Data (8 Categories)"        
    elif "2" in log_file:
        x_label_1 = "Number of Clients with Mislabeled Data (60%)"
        x_label_2 = "Number of Clients with Mislabeled Data (40%)"
        x_label_3 = "Number of Clients with Mislabeled Data (20%)"
    elif "3" in log_file:
        x_label_1 = "Number of Clients with Noisy Data (60%)"
        x_label_2 = "Number of Clients with Noisy Data (40%)"
        x_label_3 = "Number of Clients with Noisy Data (20%)"

    ab1, lb1, aa1, la1, ab2, lb2, aa2, la2, ab3, lb3, aa3, la3 = process_log(log_file)

    # x-axis as setting indices
    x = range(1, len(ab1) + 1)

    # generate plots
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))

    # use same y-axis limits for all loss plots
    all_losses = np.concatenate([lb1, la1, lb2, la2, lb3, la3])
    min_loss = all_losses.min()
    max_loss = all_losses.max()

    # accuracy and loss plots for lowest quality data setting
    make_accuracy_plot(axes[0, 0], x, ab1, aa1, x_label_1)
    make_loss_plot(axes[0, 1], x, lb1, la1, x_label_1, min_loss, max_loss)

    # accuracy and loss plots for middle quality data setting
    make_accuracy_plot(axes[1, 0], x, ab2, aa2, x_label_2)
    make_loss_plot(axes[1, 1], x, lb2, la2, x_label_2, min_loss, max_loss)

    # accuracy and loss plots for highest quality data setting
    make_accuracy_plot(axes[2, 0], x, ab3, aa3, x_label_3)
    make_loss_plot(axes[2, 1], x, lb3, la3, x_label_3, min_loss, max_loss)

    plt.tight_layout()
    plt.savefig(f"retraining/graphs/{log_file.split('/')[-1].split('.')[0]}.png")

graph_processed_log('retraining/resnet2.log')