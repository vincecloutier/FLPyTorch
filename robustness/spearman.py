import re
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np
from collections import defaultdict

def process_log(file_path):
    # read the log file
    with open(file_path, 'r') as file:
        logs = file.read()

    # regex patterns
    approx_simple_pattern = r"Banzhaf Values Simple: defaultdict\(<class 'float'>, \{(.*?)\}\)"
    approx_hessian_pattern = r"Banzhaf Values Hessian: defaultdict\(<class 'float'>, \{(.*?)\}\)"
    shapley_pattern = r"Shapley Values: defaultdict\(<class 'float'>, \{(.*?)\}\)"
    influence_pattern = r"Influence Function Values: defaultdict\(<class 'float'>, \{(.*?)\}\)"
    runtime_pattern = r"Runtimes: \{(.*?)\}"

    # extract values
    approx_simple_values = [eval(f"{{{match}}}") for match in re.findall(approx_simple_pattern, logs)]
    approx_hessian_values = [eval(f"{{{match}}}") for match in re.findall(approx_hessian_pattern, logs)]
    shapley_values = [eval(f"{{{match}}}") for match in re.findall(shapley_pattern, logs)]
    influence_values = [eval(f"{{{match}}}") for match in re.findall(influence_pattern, logs)]
    runtimes = [eval(f"{{{match}}}") for match in re.findall(runtime_pattern, logs)]

    def process_runs(runs):
        # convert list of dictionaries to df
        df = pd.DataFrame(runs)
        # sort columns based on client ids
        df = df.sort_index(axis=1)
        # handle missing values
        df = df.fillna(np.nan) 
        # rank the values within each run
        ranked_df = df.rank(axis=1, method='average', ascending=False)
        # make each run a column
        ranked_df_T = ranked_df.transpose()
        # compute spearman rank correlation
        corr_matrix = spearmanr(ranked_df_T, nan_policy='omit')[0]
        # convert to df for better readability
        corr_df = pd.DataFrame(corr_matrix, index=[f'Run {i+1}' for i in range(len(runs))], columns=[f'Run {i+1}' for i in range(len(runs))])    
        # average pairwise correlations
        upper_tri = corr_df.where(np.triu(np.ones(corr_df.shape), k=1).astype(bool))
        correlations = upper_tri.stack()
        avg_corr = correlations.mean()
        return avg_corr
    
    def process_runtimes(runtimes):
        avg_runtimes = defaultdict(float)
        for runtime in runtimes:
            for key, value in runtime.items():
                avg_runtimes[key] += value
        for key, value in avg_runtimes.items():
            avg_runtimes[key] /= len(runtimes)
        return avg_runtimes
    print(process_runtimes(runtimes))
    return process_runs(approx_simple_values), process_runs(approx_hessian_values), process_runs(shapley_values), process_runs(influence_values), process_runtimes(runtimes)


def process_and_graph_logs(log_files):
    # sum over each setting
    abvs, abvh, shapley, influence = 0, 0, 0, 0
    avg_runtimes = defaultdict(float)
    for log_file in log_files:
        approx_simple, approx_hessian, shapley, influence, runtimes = process_log(log_file)
        abvs += approx_simple
        abvh += approx_hessian
        shapley += shapley
        influence += influence
        for key, value in runtimes.items():
            avg_runtimes[key] += value
    
    # average across settings
    divisor = len(log_files)
    abvs /= divisor
    abvh /= divisor
    shapley /= divisor
    influence /= divisor
    for key, value in avg_runtimes.items():
        avg_runtimes[key] /= divisor


    # generate plots
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.bar(['ABVS', 'ABVH', 'Shapley', 'Influence'], [abvs, abvh, shapley, influence])
    plt.title(f"Spearman Rank Correlation")
    plt.xlabel("Approximation")
    plt.ylabel("Correlation")

    plt.subplot(1, 2, 2)
    plt.bar(['ABVS', 'ABVH', 'Shapley', 'Influence'], [avg_runtimes['abvs'], avg_runtimes['abvh'], avg_runtimes['sv'], avg_runtimes['if']])
    plt.title(f"Runtime")
    plt.xlabel("Approximation")
    plt.ylabel("Runtime (s)")

    plt.tight_layout()
    plt.show()


process_and_graph_logs(['robustness/cifar0.log', 'robustness/cifar1.log',  'robustness/cifar2.log', 'robustness/cifar3.log'])
