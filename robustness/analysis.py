import re
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np


def process_log(file_path):
    # Regex to match the dictionary portion { ... }
    dict_pattern = r"\{(.*?)\}"
    
    def parse_dict(dict_str):
        # This parser expects a dict string like "0: 0.1595, 1: 0.1695, ..."
        # Extract key-value pairs separated by commas
        parsed = {}
        if not dict_str.strip():
            return parsed
        entries = dict_str.split(',')
        for e in entries:
            e = e.strip()
            if e:
                kv = e.split(':', 1)
                if len(kv) == 2:
                    key, val_str = kv[0].strip(), kv[1].strip()
                    # Keys are often integers, but we can keep them as strings if needed
                    # Attempt to convert both keys and values to floats if possible
                    # If keys are always integers, we can convert them:
                    try:
                        key = int(key)
                    except ValueError:
                        pass
                    try:
                        val = float(val_str)
                    except ValueError:
                        # If for some reason it's not float, leave as string
                        val = val_str
                    parsed[key] = val
        return parsed
    
    def parse_runtime_dict(dict_str):
        # Runtimes have keys like 'abvs', 'abvh', 'sv', 'if' which are strings
        parsed = {}
        if not dict_str.strip():
            return parsed
        entries = dict_str.split(',')
        for e in entries:
            e = e.strip()
            if e:
                kv = e.split(':', 1)
                if len(kv) == 2:
                    key_str = kv[0].strip().strip("'").strip('"')
                    val_str = kv[1].strip()
                    try:
                        val = float(val_str)
                    except ValueError:
                        val = val_str
                    parsed[key_str] = val
        return parsed

    # Initialize dictionaries to hold data from each run
    abv_simple_values_dict = {}
    abv_hessian_values_dict = {}
    sv_dict = {}
    iv_dict = {}
    runtimes = {}

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            
            # Match ABV Simple
            if "ABV Simple" in line:
                # Extract the run ID
                run_id_match = re.search(r"ABV Simple (\d+):", line)
                if run_id_match:
                    run_id = int(run_id_match.group(1))
                    dict_match = re.search(dict_pattern, line)
                    if dict_match:
                        dict_str = dict_match.group(1)
                        abv_simple_values_dict[run_id] = parse_dict(dict_str)

            # Match ABV Hessian
            elif "ABV Hessian" in line:
                run_id_match = re.search(r"ABV Hessian (\d+):", line)
                if run_id_match:
                    run_id = int(run_id_match.group(1))
                    dict_match = re.search(dict_pattern, line)
                    if dict_match:
                        dict_str = dict_match.group(1)
                        abv_hessian_values_dict[run_id] = parse_dict(dict_str)

            # Match Shapley Values
            elif "Shapley Values" in line:
                run_id_match = re.search(r"Shapley Values (\d+):", line)
                if run_id_match:
                    run_id = int(run_id_match.group(1))
                    dict_match = re.search(dict_pattern, line)
                    if dict_match:
                        dict_str = dict_match.group(1)
                        sv_dict[run_id] = parse_dict(dict_str)
                    else:
                        sv_dict[run_id] = {}

            # Match Influence Values
            elif "Influence Values" in line:
                run_id_match = re.search(r"Influence Values (\d+):", line)
                if run_id_match:
                    run_id = int(run_id_match.group(1))
                    dict_match = re.search(dict_pattern, line)
                    if dict_match:
                        dict_str = dict_match.group(1)
                        iv_dict[run_id] = parse_dict(dict_str)
                    else:
                        iv_dict[run_id] = {}

            # Match Runtimes
            elif "Runtimes" in line:
                run_id_match = re.search(r"Runtimes (\d+):", line)
                if run_id_match:
                    run_id = int(run_id_match.group(1))
                    dict_match = re.search(dict_pattern, line)
                    if dict_match:
                        dict_str = dict_match.group(1)
                        runtimes[run_id] = parse_runtime_dict(dict_str)
                    else:
                        runtimes[run_id] = {}

    return abv_simple_values_dict, abv_hessian_values_dict, sv_dict, iv_dict, runtimes

    

def plot_banzhaf_boxplots(values_dict, title_prefix="Data Banzhaf"):
    # values_dict structure: {run_idx: {data_idx: value}}
    runs = list(values_dict.keys())
    if not runs:
        return  # no data to plot
    
    # Extract all data indices from the first run
    data_indices = sorted(values_dict[runs[0]].keys())

    # Gather values for each data index across all runs
    data_by_index = []
    for di in data_indices:
        vals = [values_dict[ri][di] for ri in runs]
        data_by_index.append(vals)
    
    # compute median across runs for each data index
    medians = [np.median(vals) for vals in data_by_index]
    # sort data indices by median
    sorted_indices = np.argsort(medians)
    
    # Reorder data by index according to the median ranking
    data_by_index_sorted = [data_by_index[i] for i in sorted_indices]
    sorted_medians = [medians[i] for i in sorted_indices]

    # Compute baseline ranking: this is just the sorted order of medians
    # baseline_ranks = ranks of data_indices after sorting by median
    # We'll use the sorted order as the "true" ranking to compare against
    # The baseline ranking (by median) for each data_index is given by their position in sorted_indices
    baseline_ranks = np.arange(len(data_indices))  # baseline ranks from 0 to len-1
    
    # Compute Spearman correlations for each run against the baseline ranking
    spearman_values = []
    for ri in runs:
        # Extract this run's values in the order of the original data_indices
        run_values = [values_dict[ri][di] for di in data_indices]
        # Get the order of data_indices for this run
        run_sorted_order = np.argsort(run_values)
        
        # Now we have run_sorted_order which gives the indices of data_indices in ascending order for run ri
        # We need to map these indices to ranks
        run_ranks = np.empty_like(run_sorted_order)
        run_ranks[run_sorted_order] = np.arange(len(run_values))
        
        # Spearman correlation between run_ranks and baseline_ranks
        # Note: baseline_ranks corresponds to sorted_indices order. We need to align them.
        # baseline_ranks currently is just an array [0,1,2,...] representing ranks after median sorting.
        # data_indices_sorted (via sorted_indices) gives the order of indices by median.
        # run_ranks is indexed by the position in the original data_indices.
        # We must apply the same permutation to run_ranks that we used to get median order.
        
        # sorted_indices gives the mapping from median order to original order.
        # We want run_ranks in median-sorted order:
        run_ranks_in_median_order = run_ranks[sorted_indices]
        
        # Now we compute spearman correlation between run_ranks_in_median_order and baseline_ranks
        sp, _ = spearmanr(run_ranks_in_median_order, baseline_ranks)
        spearman_values.append(sp)
    
    # Average Spearman correlation across runs
    return np.mean(spearman_values)

def process_and_graph_logs(log_file):
    abv_simple_values_dict, abv_hessian_values_dict, shapley_values, influence_values, runtimes = process_log(log_file)

    # Plot the FBV Simple values
    print(log_file)
    print(plot_banzhaf_boxplots(abv_simple_values_dict, title_prefix="Data Banzhaf (FBV-Simple)"))
    print(plot_banzhaf_boxplots(abv_hessian_values_dict, title_prefix="Data Banzhaf (FBV-Hessian)"))
    # print(plot_banzhaf_boxplots(shapley_values, title_prefix="Data Banzhaf (Shapley)"))
    print(plot_banzhaf_boxplots(influence_values, title_prefix="Data Banzhaf (Influence)"))
    # print(plot_banzhaf_boxplots(runtimes, title_prefix="Data Banzhaf (Runtimes)"))


def process_and_graph_logs(log_files, plot=False):
    # lists to hold rank stability per setting
    corr_metrics = {'FBVS': [], 'FBVH': []}

    # Since we no longer have shapley or influence from these logs, we won't compute them.
    # If you need them later, you can add them back similarly.

    # No runtimes in the given logs (as per original snippet), so skip them or set empty.
    runtime_metrics = {'FBVS': [], 'FBVH': []}

    approx_simple, approx_hessian, shapley, influence, runtimes = process_log(log_file)


    # compute summary statistics across settings
    methods = ["FBVS", "FBVH"]
    avg_corrs = [np.mean(corr_metrics[m]) for m in methods]
    print(avg_corrs)

    if plot:
        # generate plots
        # Attempt to extract dataset name from the first file name
        match = re.search(r'/(.+)\d', log_files[0])
        if match:
            dataset = match.group(1)
        else:
            dataset = "dataset"

        colors = plt.get_cmap('tab10').colors
        # only two methods now, select two colors
        colors = [colors[0], colors[2]]

        plt.figure(figsize=(8, 4.5), layout="constrained")
        plt.subplot(1, 2, 1)
        
        plt.bar(methods, avg_corrs, capsize=5, color=colors)
        for i, mean in enumerate(avg_corrs):
            plt.text(i, mean, f'{mean:.2f}', ha='center', va='bottom')
        plt.title("Average Spearman Rank Correlation")
        plt.xlabel("Data Valuation Method")
        plt.ylim(0, 1)
        plt.ylabel("Correlation")

        # If you have no runtimes, you can skip the second plot
        # If you want to show an empty runtime plot or some placeholder:
        plt.subplot(1, 2, 2)
        # Just as a placeholder, we'll show no data
        plt.bar([], [])
        plt.title("No Runtimes Available")
        plt.xlabel("Data Valuation Method")
        plt.ylabel("Runtime (s)")

        plt.savefig(f"robustness/graphs/robustness_{dataset}.png", dpi=300, bbox_inches='tight')

# # Example usage:
process_and_graph_logs('robustness2/data.log', plot=True)