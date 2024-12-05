import re
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
import numpy as np

def process_and_graph_log(file_path, plot=True):
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

    print(approx_simple_values)
    print(approx_hessian_values)
    print(shapley_values)
    print(influence_values)
    print(runtimes)

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

    print(process_runs(approx_simple_values))
    print(process_runs(approx_hessian_values))
    print(process_runs(shapley_values))
    print(process_runs(influence_values))


    # if plot:
    #     # Plotting
    #     plt.figure(figsize=(10, 6))

    #     # Runtimes plot
    #     plt.subplot(1, 2, 1)
    #     runtime_means = runtime_df.mean()
    #     runtime_means.plot(kind="bar", title="Average Runtimes of Methods", rot=45)
    #     plt.ylabel("Runtime (seconds)")

    #     # correlation bar graph
    #     plt.subplot(1, 2, 2)


    #     plt.tight_layout()
    #     plt.show()


# Run the function on the example log file
process_and_graph_log('robustness/cifar3.log', plot=True)