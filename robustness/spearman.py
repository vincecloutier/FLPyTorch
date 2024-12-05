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

    return process_runs(approx_simple_values), process_runs(approx_hessian_values), process_runs(shapley_values), process_runs(influence_values), process_runtimes(runtimes)


def process_and_graph_logs(log_files):
    # collect data from each setting
    abvs, abvh, sv, iv = [], [], [], []
    avg_runtimes = defaultdict(float)
    for log_file in log_files:
        approx_simple, approx_hessian, shapley, influence, runtimes = process_log(log_file)
        abvs.append(approx_simple)
        abvh.append(approx_hessian)
        sv.append(shapley)
        iv.append(influence)
        for key, value in runtimes.items():
            avg_runtimes[key] += value
    
    # average across settings
    abvs = np.mean(abvs, axis=0)
    abvh = np.mean(abvh, axis=0)
    sv = np.mean(sv, axis=0)
    iv = np.mean(iv, axis=0)
    for key, value in avg_runtimes.items():
        avg_runtimes[key] /= len(log_files)

    # generate plots
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.bar(['ABVS', 'ABVH', 'Shapley', 'Influence'], [abvs, abvh, sv, iv])
    for i, v in enumerate([abvs, abvh, sv, iv]):
        plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    plt.title(f"Spearman Rank Correlation")
    plt.xlabel("Approximation")
    plt.ylabel("Correlation")

    plt.subplot(1, 2, 2)
    plt.bar(['ABVS', 'ABVH', 'Shapley', 'Influence'], [avg_runtimes['abvs'], avg_runtimes['abvh'], avg_runtimes['sv'], avg_runtimes['if']])
    for i, v in enumerate([avg_runtimes['abvs'], avg_runtimes['abvh'], avg_runtimes['sv'], avg_runtimes['if']]):
        plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    plt.yscale('log')
    plt.title(f"Runtime")
    plt.xlabel("Approximation")
    plt.ylabel("Runtime (s)")

    plt.tight_layout()
    plt.show()

process_and_graph_logs(['robustness/cifar0.log', 'robustness/cifar1.log',  'robustness/cifar2.log', 'robustness/cifar3.log'])
