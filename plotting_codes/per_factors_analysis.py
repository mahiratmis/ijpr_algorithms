import ast
import pandas as pd
import csv
import itertools
import numpy as np

# Open the CSV
df = pd.read_csv('../results/all_results_with_meta.csv')
# print df.info()

# get unique algorithm types
algorithm_types = df["algorithm_type"].unique()

# slice first 128 rows, and columns from n_sku to cost_ci
factors_df = df.loc[:127, 'n_sku':'cost_ci']
factor_names = factors_df.columns.tolist()


# Estimate per factor runtime average and total cost average
# in results, cols will be the algorithm types and row indices 
# will be factor names and their values combined 
# cells will contain average  running time or total cost for that factor
run_time_avgs = {}
benchmark_name_factor_mean = []
means_per_factor = {}
for stats_col in ["running_time", "total_cost"]:
    for alg_type in algorithm_types:
        # we dont have stats for them
        if alg_type in ["Flexible", "Dedicated"] and stats_col=="running_time":
            continue
        means_per_factor[alg_type] = []
        means_per_factor["factor_names"] =[]
        for factor_name in factor_names:
            factor_name_unique_vals = df[factor_name].unique()
            for fac_val in factor_name_unique_vals:
                # get mean value of the stats_col column for the rows that contain fac_val in factor_name column and has
                # alg_type value in algorithm_type column
                mean = df.loc[(df[factor_name] == fac_val) & (df["algorithm_type"] == alg_type), stats_col].mean()
                if not isinstance(fac_val, str) and fac_val < 1:
                    fac_val = int(fac_val * 100)  # handle decimal values
                name = factor_name + "_" + str(fac_val)
                means_per_factor["factor_names"].append(name)
                means_per_factor[alg_type].append(round(mean,2))
    pd.DataFrame(means_per_factor).to_csv (f"perfactor_{stats_col}.csv", index = None, header=True)
    

print (algorithm_types)
# get binary combinations of algorithms
combinations = list(itertools.combinations(algorithm_types, r=2))

# perfactor % difference of total cost and run times
diff_percentages={}
for stats_col in ["running_time", "total_cost"]:
    df_means = pd.read_csv(f"perfactor_{stats_col}.csv")
    for alg1, alg2 in combinations:
        # we dont have running time stats for "Flexible" and "Dedicated"
        if (alg1 in ["Flexible", "Dedicated"] and stats_col=="running_time") or \
           (alg2 in ["Flexible", "Dedicated"] and stats_col=="running_time"):
            continue
        means1 = df_means[alg1].values
        means2 = df_means[alg2].values
        diff_percentages[f"{alg1}_{alg2}"] = (means1 - means2) * 100.0 / means1
    pd.DataFrame(diff_percentages).to_csv (f"perfactor_percent_{stats_col}.csv", index = None, header=True)


# total cost values for algorithms as columns cases ids as index
total_cost = {}
stats_col = "total_cost"
for alg in algorithm_types:
    total_cost[alg] = np.round(df[df["algorithm_type"] == alg][stats_col].values,2)
# save result
case_based_total_costs_df = pd.DataFrame(total_cost, index=range(1,129))
case_based_total_costs_df.to_csv("case_based_total_costs.csv")

# check which algorithms have minimum cost values
case_based_total_costs_np = case_based_total_costs_df.values
case_based_min_tot_cost_vals = case_based_total_costs_np.min(axis=1)
res = (case_based_total_costs_np == case_based_min_tot_cost_vals[:,None]).astype(int) # add extra dimension to broadcast
case_based_min_total_cost_mask_df = pd.DataFrame(res, columns=case_based_total_costs_df.columns, index=range(1,129))
case_based_min_total_cost_mask_df.to_csv("case_based_min_total_cost_mask.csv")


# calculate total cost difference per algorithm pair
total_cost_benchmarks = {}
stats_col = "total_cost"
for alg1, alg2 in combinations:
    tc_first = df[df["algorithm_type"] == alg1][stats_col].values  # to numpy
    tc_second = df[df["algorithm_type"] == alg2][stats_col].values
    total_cost_percentage = (tc_first - tc_second) * 100.0 / tc_first
    total_cost_benchmarks[alg1 + "_" + alg2] = np.round(total_cost_percentage,2)
# columns benchmark names, rows total cost % performance
total_cost_diffs_df = pd.DataFrame(total_cost_benchmarks, index=range(1,129))
# save intermediate result
total_cost_diffs_df.to_csv("case_based_total_cost_diffs_df.csv")


# calculate cost diffs for bbox plot
cost_diffs = {}
for stats_col in ["total_cost"]:
    total_cost_benchmarks.clear()
    for alg1, alg2 in combinations:
        if "MSSA" not in [alg1, alg2]:
            continue
        # make sure the first one is mssamean
        if alg2 == "MSSA":
            alg1, alg2 = alg2, alg1
        for factor_name in factor_names:
            factor_name_unique_vals = df[factor_name].unique()
            for fac_val in factor_name_unique_vals:
                vals_mtsa = df.loc[(df[factor_name] == fac_val) & (df["algorithm_type"] == alg1), stats_col].values
                vals_other = df.loc[(df[factor_name] == fac_val) & (df["algorithm_type"] == alg2), stats_col].values
                assert len(vals_mtsa) == 64
                if not isinstance(fac_val, str) and fac_val < 1:
                    fac_val = int(fac_val*100)  # handle decimal values
                cost_diffs[f"{alg2}_{factor_name}_{str(fac_val)}"] = -np.round((vals_mtsa - vals_other) * 100.0 / vals_mtsa,2)

    cost_diffs_df = pd.DataFrame(cost_diffs)
    # save new statistics as a csv file
    cost_diffs_df.to_csv("bbox_"+stats_col + "_benchmark.csv") 
