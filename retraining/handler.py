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


def make_accuracy_plot(ax, x, acc_before, acc_after):
    ax.plot(x, acc_before, marker='o', linestyle='-', color='tab:orange', label='Initial Accuracy')
    ax.plot(x, acc_after, marker='s', linestyle='--', color='tab:blue', label='Retrained Accuracy')
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}" for n in x], fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)

def make_loss_plot(ax, x, loss_before, loss_after, min_loss, max_loss):
    ax.plot(x, loss_before, marker='o', linestyle='-', color='tab:orange', label='Initial Loss')
    ax.plot(x, loss_after, marker='s', linestyle='--', color='tab:blue', label='Retrained Loss')
    ax.set_ylabel("Loss", fontsize=12)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_ylim(min_loss * 0.95, max_loss * 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)

def graph_processed_log(log_file):
    # determine x-axis labels based on the log file identifier
    if "1" in log_file:
        title = "Accuracy and Loss Before and After Retraining (Non-IID Data)"
        x_labels = [
            "Number of Clients with Only 4 Categories Of Data",
            "Number of Clients with Only 6 Categories Of Data",
            "Number of Clients with Only 8 Categories Of Data"
        ]
    elif "2" in log_file:
        title = "Accuracy and Loss Before and After Retraining (Mislabelled Data)"
        x_labels = [
            "Number of Clients with Mislabelled Sample Ratio 60%",
            "Number of Clients with Mislabelled Sample Ratio 40%",
            "Number of Clients with Mislabelled Sample Ratio 20%"
        ]
    elif "3" in log_file:
        title = "Accuracy and Loss Before and After Retraining (Poisoned Data)"
        x_labels = [
            "Number of Clients with Poison Sample Ratio 60%",
            "Number of Clients with Poison Sample Ratio 40%",
            "Number of Clients with Poison Sample Ratio 20%"
        ]
    else:
        x_labels = ["Setting 1", "Setting 2", "Setting 3"]  # Default labels if none match

    # process the log file to extract necessary data
    ab1, lb1, aa1, la1, ab2, lb2, aa2, la2, ab3, lb3, aa3, la3 = process_log(log_file)

    # define x-axis values based on the number of data points
    x = range(1, len(ab1) + 1)

    # create the main figure
    fig = plt.figure(figsize=(8, 10), layout="constrained") 

    # create 3 subfigures, each representing a row
    subfigs = fig.subfigures(nrows=3, ncols=1,)

    # determine the minimum and maximum loss across all settings for consistent y-axis limits
    all_losses = np.concatenate([lb1, la1, lb2, la2, lb3, la3])
    min_loss = all_losses.min()
    max_loss = all_losses.max()

    # list of all settings to iterate through
    settings = [
        (ab1, aa1, lb1, la1, x_labels[0]),
        (ab2, aa2, lb2, la2, x_labels[1]),
        (ab3, aa3, lb3, la3, x_labels[2]),
    ]

    for subfig, (ab, aa, lb, la, x_label) in zip(subfigs, settings):
        # create two subplots within each subfigure
        axs = subfig.subplots(1, 2, sharey=False, sharex=True)
        # generate accuracy and loss plots
        make_accuracy_plot(axs[0], x, ab, aa)
        make_loss_plot(axs[1], x, lb, la, min_loss, max_loss)
        # add a super x-label for the subfigure
        subfig.supxlabel(x_label, fontsize=12)
  
    # add a main title for the entire figure
    fig.suptitle(title, fontsize=14)

    # save the figure to the specified directory with the log file's base name
    plt.savefig(f"retraining/graphs/{log_file.split('/')[-1].split('.')[0]}.png", dpi=300)

graph_processed_log('retraining/resnet1.log')
graph_processed_log('retraining/resnet2.log')
graph_processed_log('retraining/resnet3.log')