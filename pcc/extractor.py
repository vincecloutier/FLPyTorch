import re
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def process_and_graph_log(file_path):
    # Read the log file
    with open(file_path, 'r') as file:
        logs = file.read()

    # regex patterns to extract data
    shapley_pattern = r"Shapley Values: \[([-\d.,\s]+)\]"
    approx_simple_pattern = r"Approximate Banzhaf Values Simple: \[([-\d.,\s]+)\]"
    approx_hessian_pattern = r"Approximate Banzhaf Values Hessian: \[([-\d.,\s]+)\]"

    # extract values
    shapley_values = [list(map(float, match.split(','))) for match in re.findall(shapley_pattern, logs)]
    approx_simple_values = [list(map(float, match.split(','))) for match in re.findall(approx_simple_pattern, logs)]
    approx_hessian_values = [list(map(float, match.split(','))) for match in re.findall(approx_hessian_pattern, logs)]

    # concatenate across runs
    shapley_all = [val for run in shapley_values for val in run]
    approx_simple_all = [val for run in approx_simple_values for val in run]
    approx_hessian_all = [val for run in approx_hessian_values for val in run]

    # min_max scale the data to 0-1
    scaler = MinMaxScaler()
    shapley_all = scaler.fit_transform(np.array(shapley_all).reshape(-1, 1)).flatten()
    approx_simple_all = scaler.fit_transform(np.array(approx_simple_all).reshape(-1, 1)).flatten()
    approx_hessian_all = scaler.fit_transform(np.array(approx_hessian_all).reshape(-1, 1)).flatten()

    # create a dataframe for easier processing
    data = pd.DataFrame({
        'Shapley': shapley_all,
        'Approx_Simple': approx_simple_all,
        'Approx_Hessian': approx_hessian_all
    })

    # calculate pearson correlation coefficients
    corr_shapley_simple = pearsonr(data['Shapley'], data['Approx_Simple'])[0]
    corr_shapley_hessian = pearsonr(data['Shapley'], data['Approx_Hessian'])[0]

    # print correlations
    print("Pearson Correlation Coefficients:")
    print(f"Shapley and Approx Simple: {corr_shapley_simple:.4f}")
    print(f"Shapley and Approx Hessian: {corr_shapley_hessian:.4f}")

    # generate plots
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.scatter(data['Shapley'], data['Approx_Simple'], alpha=0.6)
    plt.title(f"Shapley vs Approx Simple (Corr: {corr_shapley_simple:.2f})")
    plt.xlabel("Shapley")
    plt.ylabel("Approx Simple")

    plt.subplot(1, 2, 2)
    plt.scatter(data['Shapley'], data['Approx_Hessian'], alpha=0.6)
    plt.title(f"Shapley vs Approx Hessian (Corr: {corr_shapley_hessian:.2f})")
    plt.xlabel("Shapley")
    plt.ylabel("Approx Hessian")
    
    plt.tight_layout()
    plt.show()

# example usage
process_and_graph_log('pcc/cifarconvergence.log')
