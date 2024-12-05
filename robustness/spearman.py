import re
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from collections import defaultdict
import glob

# function to parse a single run's log content
def parse_run_log(log_content):
    # regex patterns to extract data
    approx_simple_pattern = r"Banzhaf Values Simple: defaultdict\(<class 'float'>, \{(.*?)\}\)"
    approx_hessian_pattern = r"Banzhaf Values Hessian: defaultdict\(<class 'float'>, \{(.*?)\}\)"
    shapley_pattern = r"Shapley Values: defaultdict\(<class 'float'>, \{(.*?)\}\)"
    influence_pattern = r"Influence Function Values: defaultdict\(<class 'float'>, \{(.*?)\}\)"
    runtime_pattern = r"Runtimes: \{(.*?)\}"

    # function to parse defaultdict format
    def parse_defaultdict(text):
        # extract key-value pairs
        pairs = re.findall(r"(\d+): ([\-0-9.]+)", text)
        return {int(k): float(v) for k, v in pairs}

    # extract values
    approx_simple_match = re.search(approx_simple_pattern, log_content)
    approx_hessian_match = re.search(approx_hessian_pattern, log_content)
    shapley_match = re.search(shapley_pattern, log_content)
    influence_match = re.search(influence_pattern, log_content)
    runtime_match = re.search(runtime_pattern, log_content)

    approx_simple = parse_defaultdict(approx_simple_match.group(1)) if approx_simple_match else {}
    approx_hessian = parse_defaultdict(approx_hessian_match.group(1)) if approx_hessian_match else {}
    shapley = parse_defaultdict(shapley_match.group(1)) if shapley_match else {}
    influence = parse_defaultdict(influence_match.group(1)) if influence_match else {}
    runtimes = parse_defaultdict(runtime_match.group(1)) if runtime_match else {}

    return {
        "Banzhaf_Simple": approx_simple,
        "Banzhaf_Hessian": approx_hessian,
        "Shapley": shapley,
        "Influence": influence,
        "Runtimes": runtimes
    }

# function to compute spearman correlations for a method across runs
def compute_spearman(method_rankings):
    correlations = []
    num_runs = len(method_rankings)
    for i in range(num_runs):
        for j in range(i+1, num_runs):
            rank1 = method_rankings[i]
            rank2 = method_rankings[j]
            # Ensure the rankings have the same keys
            common_keys = set(rank1.keys()).intersection(set(rank2.keys()))
            if len(common_keys) < 2:
                # Spearman correlation is not defined for less than 2 points
                continue
            ordered_rank1 = [rank1[k] for k in common_keys]
            ordered_rank2 = [rank2[k] for k in common_keys]
            corr, _ = spearmanr(ordered_rank1, ordered_rank2)
            if not pd.isna(corr):
                correlations.append(corr)
    if correlations:
        return sum(correlations) / len(correlations)
    else:
        return 0


all_runs_data = []


with open("robustness/cifar2.log", 'r') as file:
    content = file.read()
    run_sections = re.split(r"INFO Run \d+: Adding Gaussian noise with std=.*", content)
    for section in run_sections[1:]:  # first split is before the first run
        run_data = parse_run_log(section)
        print(run_data)
        all_runs_data.append(run_data)

# initialize dictionaries to hold rankings and runtimes
methods = ["Banzhaf_Simple", "Banzhaf_Hessian", "Shapley", "Influence"]
rankings = {method: [] for method in methods}
runtimes = {method: [] for method in methods}

for run in all_runs_data:
    # rank clients for each method (higher value gets higher rank)
    for method in methods:
        method_values = run.get(method, {})
        if method_values:
            # sort clients based on valuation, higher first
            sorted_clients = sorted(method_values.items(), key=lambda item: item[1], reverse=True)
            # assign ranks starting from 1
            method_rank = {client: rank for rank, (client, _) in enumerate(sorted_clients, start=1)}
            rankings[method].append(method_rank)
    # collect runtimes
    run_runtimes = run.get("Runtimes", {})
    for runtime_method, runtime_value in run_runtimes.items():
        runtimes[runtime_method].append(runtime_value)

# compute spearman rank correlations for each method
average_spearman = {}
for method in methods:
    method_rankings = rankings[method]
    if len(method_rankings) < 2:
        average_spearman[method] = 0
    else:
        avg_corr = compute_spearman(method_rankings)
        average_spearman[method] = avg_corr

# print average spearman correlations
print("Average Spearman Rank Correlation Coefficients Across Runs:")
for method, avg_corr in average_spearman.items():
    print(f"{method}: {avg_corr:.4f}")

# compute average runtimes for each valuation method
average_runtimes = {}
for method, runtime_list in runtimes.items():
    if runtime_list:
        average_runtimes[method] = sum(runtime_list) / len(runtime_list)
    else:
        average_runtimes[method] = 0

# print average runtimes
print("\nAverage Runtimes (in seconds) Across Runs:")
for method, avg_rt in average_runtimes.items():
    print(f"{method}: {avg_rt:.4f}")

# plotting spearman rank correlation coefficients
plt.figure(figsize=(12, 6))

# spearman correlation plot
plt.subplot(1, 2, 1)
methods_labels = list(average_spearman.keys())
spearman_values = list(average_spearman.values())
plt.bar(methods_labels, spearman_values, color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Data Valuation Methods')
plt.ylabel('Average Spearman Rank Correlation')
plt.title('Ranking Stability Across Runs')
plt.ylim(0, 1)
plt.grid(axis='y')

# Runtime Comparison Plot
plt.subplot(1, 2, 2)
runtime_methods = list(average_runtimes.keys())
runtime_values = list(average_runtimes.values())
plt.bar(runtime_methods, runtime_values, color=['cyan', 'lime', 'orange', 'magenta', 'yellow'])
plt.xlabel('Data Valuation Methods')
plt.ylabel('Average Runtime (seconds)')
plt.title('Comparative Runtimes Across Runs')
plt.grid(axis='y')

plt.tight_layout()
plt.show()