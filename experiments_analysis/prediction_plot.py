import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import shutil
from matplotlib.ticker import PercentFormatter, MaxNLocator

from experiments_analysis.experiment_plot import plot_linear_for_multiple_qps

experiment_name_replacement = {"min latency": "Block", "min infass load": "INFaaS++",
                               "request per seconds": "Instance-QPM"}
scheduler_name_ordered = ['Round Robin', 'random', 'INFaaS++', 'Instance-QPM', 'Block*', 'Block']


def directory_name_parser(directory_name):
    directory_name = directory_name.split("_")
    qps = directory_name[1]
    n = directory_name[6]
    chunked = directory_name[8]
    return qps, n, chunked


def extract_prediction_errors(experiment):
    average_prediction_errors_ratio = experiment['prediction_errors']
    compare_error_rate = experiment['compare_error_rate']
    return average_prediction_errors_ratio, compare_error_rate


def plot_prediction_hit_histogram(ax, selected_ranks,
                                  x_label,
                                  y_label,
                                  title,
                                  enable_x_label=False,
                                  enable_y_label=False,
                                  enable_title=False,
                                  title_fontsize=10):
    ax.hist(selected_ranks, bins=np.arange(1, max(selected_ranks)), edgecolor='black', alpha=0.7,
            weights=np.ones_like(selected_ranks) / len(selected_ranks))
    ax.set_xlim(1, max(selected_ranks))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if enable_x_label:
        ax.set_xlabel(x_label, fontsize=title_fontsize)
    if enable_y_label:
        ax.set_ylabel(y_label, fontsize=title_fontsize)
    if enable_title:
        ax.set_title(title, fontsize=title_fontsize, loc='center')


def plot_prediction_hit_histogram_for_multiple_qps(axes, rank_data, x_label, y_label, title,
                                                   enable_middle_x_label=False,
                                                   enable_middle_title=False,
                                                   title_fontsize=10):
    enable_y_label = True
    for qps in rank_data.keys():
        ax = axes.get(qps)
        selected_ranks = rank_data[qps]
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylim(0, 1)
        for feature in selected_ranks.keys():
            selected_rank = selected_ranks[feature]
            enable_title = enable_middle_title and (qps == list(rank_data.keys())[len(rank_data) // 2])
            enable_x_label = enable_middle_x_label and (qps == list(rank_data.keys())[len(rank_data) // 2])
            plot_prediction_hit_histogram(ax, selected_rank, x_label, y_label, title,
                                          enable_x_label=enable_x_label,
                                          enable_y_label=enable_y_label,
                                          enable_title=enable_title,
                                          title_fontsize=title_fontsize)
            enable_y_label = False  # Disable y-label for subsequent plots


def plot_line_scatter_mixed(ax, scatter_y, linea_y):
    """
    Plot a line and scatter mixed plot on the given axes.
    :param ax: The axes to plot on.
    :param scatter_y: The y-values for the scatter plot.
    :param linea_y: The y-values for the line plot.
    """
    assert len(scatter_y) == len(linea_y), "Scatter and line data must have the same length"
    # scatter is a 2 dimensional np array, x is the index of first column
    # ax.scatter(x, scatter_y, color='blue', label='Scatter Data', s=10)
    # ax.plot(x, linea_y, color='red', label='Line Data')
    for key in scatter_y.keys():
        x = np.arange(len(scatter_y[key]))
        for i in range(scatter_y[key].shape[1]):
            if i == 0:
                ax.scatter(x, scatter_y[key][:, i], color='blue', s=10, alpha=0.15, label='Sampled Real Serving Latency')
            else:
                ax.scatter(x, scatter_y[key][:, i], color='blue', s=10, alpha=0.15)
            ax.fill_between(x, np.min(scatter_y[key], axis=1), np.max(scatter_y[key], axis=1),
                            color='blue', alpha=0.01)
        ax.plot(x, np.mean(scatter_y[key], axis=1), color='black', label='Mean Serving Latency')
        ax.plot(x, linea_y[key], color='red', label='Selected Instance Latency')
        # ax.plot(x, np.min(scatter_y[key], axis=1), color='black', label='Min Serving Latency')
        # ax.plot(x, np.max(scatter_y[key], axis=1), color='orange', label='Max Serving Latency')


def plot_linea_scatter_for_multiple_qps(axes, linear_data, scatter_data, x_label="Sampled Query ID",
                                        legend_anchor=(1.05, 1.3),
                                        metric_name="Real Latency (s)",
                                        enable_legend_at_middle=True,
                                        enable_middle_x_label=False,
                                        enable_x_label_at_left_corner=False,
                                        x_label_coords=(-0.3, -0.105),
                                        title_fontsize=10):
    """
    Plot multiple line and scatter mixed plots for different QPS values.
    """
    assert len(scatter_data) == len(linear_data), "Scatter and line data must have the same length"
    enable_label = True
    i = 0
    for qps in linear_data.keys():
        ax = axes.get(qps)
        qps_linear_data = linear_data[qps]
        qps_scatter_data = scatter_data[qps]
        plot_line_scatter_mixed(ax, qps_scatter_data, qps_linear_data)
        if enable_label:
            ax.set_ylabel(f"{metric_name}", fontsize=title_fontsize)
            enable_label = False

        if enable_legend_at_middle and i == len(linear_data) // 2:
            ax.legend(fancybox=False, shadow=False, ncol=6, fontsize=title_fontsize,
                      loc='upper right', bbox_to_anchor=legend_anchor)
        if enable_middle_x_label and i == len(linear_data) // 2:
            ax.set_xlabel(x_label, fontsize=title_fontsize)
        elif enable_x_label_at_left_corner and i == 0:
            ax.set_xlabel(x_label, fontsize=title_fontsize)
            ax.xaxis.set_label_coords(*x_label_coords)
        i += 1


def plot_per_qps(experiments_set, output_dir, min_qps=1, max_qps=36):
    qps_output_dir = output_dir
    if os.path.exists(qps_output_dir):
        shutil.rmtree(qps_output_dir)
    os.makedirs(qps_output_dir)

    # avg_free_gpu = {}
    # var_free_gpu_per_node = {}
    # num_total_preemption = {}

    qps_set = sorted(set([record["qps"] for record in experiments_set]))
    if min_qps > 0:
        qps_set = [qps for qps in qps_set if min_qps <= qps <= max_qps]

    sorted_keys = []
    map_from_name_exp = {}
    prediction_overhead = {}
    prediction_overhead_ratio = {}
    prediction_errors = {}
    compare_errors = {}
    prediction_errors_rate = {}
    sampled_predict_latency = {}
    sampled_serving_latencies = {}
    sampled_rank_bucket = {}
    for qps in qps_set:
        prediction_overhead_per_qps = {}
        prediction_overhead_ratio_per_qps = {}
        prediction_errors_per_qps = {}
        prediction_errors_rate_per_qps = {}
        compare_errors_per_qps = {}
        sampled_predict_latency_per_qps = {}
        sampled_serving_latencies_per_qps = {}
        sampled_rank_bucket_per_qps = {}
        prediction_overhead[qps] = prediction_overhead_per_qps
        prediction_overhead_ratio[qps] = prediction_overhead_ratio_per_qps
        prediction_errors[qps] = prediction_errors_per_qps
        prediction_errors_rate[qps] = prediction_errors_rate_per_qps
        compare_errors[qps] = compare_errors_per_qps
        sampled_predict_latency[qps] = sampled_predict_latency_per_qps
        sampled_serving_latencies[qps] = sampled_serving_latencies_per_qps
        sampled_rank_bucket[qps] = sampled_rank_bucket_per_qps

        qps_experiments = [record for record in experiments_set if record["qps"] == qps]
        for experiment in qps_experiments:
            if experiment['chunked'] == 'true':
                experiment_name = "Chunked Prefilled"
            else:
                experiment_name = "Prefilled Prioritized"
            for key in experiment_name_replacement.keys():
                if key in experiment_name:
                    experiment_name = experiment_name.replace(key, experiment_name_replacement[key])
            map_from_name_exp[experiment_name] = experiment
        ordered_key = []
        if len(sorted_keys) == 0:
            sorted_keys = sorted(map_from_name_exp.keys())
            for key in scheduler_name_ordered:
                if key in sorted_keys:
                    sorted_keys.remove(key)
                    ordered_key.append(key)
            sorted_keys = sorted_keys + ordered_key
        for index_name in sorted_keys:
            experiments = map_from_name_exp[index_name]
            if index_name not in prediction_overhead:
                current_prediction_overhead = experiments['prediction_overhead']
                prediction_overhead_per_qps[index_name] = current_prediction_overhead
                end_to_end_latencies = experiments['request_latencies']
                prediction_overhead_ratio_per_qps[index_name] = [(100.0 * overhead / latency)
                                                                 for overhead, latency
                                                                 in
                                                                 zip(current_prediction_overhead, end_to_end_latencies)
                                                                 if latency > 0]
                prediction_errors_per_qps[index_name] = experiments['prediction_errors']
                compare_errors_per_qps[index_name] = experiments['compare_error_rate']
                if index_name == "Chunked Prefilled":
                    sampled_predict_latency_per_qps[index_name] = experiments['min_predicted_latency']
                    sampled_serving_latencies_per_qps[index_name] = experiments['serving_latencies']
                    sampled_rank_bucket_per_qps[index_name] = experiments['sampled_selected_instance_rank']

    fig, axs = plt.subplots(3, len(qps_set))
    axs_for_prediction_overhead_ratio = {}
    axs_for_prediction_errors_rate = {}
    axs_for_selected_instance_rank_bucket = {}

    for i, qps_value in enumerate(qps_set):
        # axs_for_prediction_overhead[qps_value] = axs[0][i]
        axs_for_prediction_errors_rate[qps_value] = axs[0][i]
        axs_for_prediction_overhead_ratio[qps_value] = axs[1][i]
        axs_for_selected_instance_rank_bucket[qps_value] = axs[2][i]

    # plot_linear_for_multiple_qps(axs_for_prediction_overhead, prediction_overhead, "Overhead (ms)",
    #                              sigma=10,
    #                              enable_legend_at_middle=True, legend_anchor=(1.1, 1.25),
    #                              title_fontsize=10)
    plot_linear_for_multiple_qps(axs_for_prediction_errors_rate, prediction_errors,
                                 "Error Rate", sigma=1,
                                 enable_legend_at_middle=True, legend_anchor=(1.0, 1.25),
                                 title_fontsize=10,
                                 enable_title_labels=True,
                                 enable_x_label_at_left_corner=True,
                                 x_label="Sample ID: ",
                                 x_label_coords=(-0.32, -0.105),
                                 max_num_samples=100)

    plot_linea_scatter_for_multiple_qps(axs_for_prediction_overhead_ratio,
                                        sampled_predict_latency,
                                        sampled_serving_latencies,
                                        enable_x_label_at_left_corner=True,
                                        legend_anchor=(1.8, 1.15),
                                        x_label="Sample ID: ",
                                        x_label_coords=(-0.15, -0.105))

    plot_prediction_hit_histogram_for_multiple_qps(axs_for_selected_instance_rank_bucket,
                                                   sampled_rank_bucket,
                                                   "The Selected Instance's Real Latency Position",
                                                   "Percentage",
                                                   "Distribution",
                                                   enable_middle_x_label=True,
                                                   enable_middle_title=True,
                                                   title_fontsize=11)

    fig.tight_layout()
    fig.set_size_inches(18, 10)
    fig.subplots_adjust(hspace=0.25, wspace=0.28)
    plt.savefig(qps_output_dir + "/profiling.png", bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser(description='Plot the results of the experiments')
    parser.add_argument("--experiments-dir", type=str,
                        default="experiments_analysis/single_node_experiment_output/sharegpt")
    parser.add_argument("--output-dir", type=str, default="./experiments_analysis/single_node_exp_plots")
    parser.add_argument("--plot-per-qps", type=bool, default=True)
    # parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    data_dir = os.getcwd() + "/" + args.experiments_dir

    experiments_set = []
    for scheduler_name in os.listdir(data_dir):
        scheduler_dir = data_dir + "/" + scheduler_name
        if scheduler_name == 'logs':
            continue
        for directory in os.listdir(scheduler_dir):
            record = {"scheduler_name": "block"}
            experiments_set.append(record)
            qps, n, chunked = directory_name_parser(directory)
            record["qps"] = float(qps)
            record["n"] = int(n)
            record["chunked"] = chunked
            for experiments_trace in os.listdir(scheduler_dir + "/" + directory):
                if experiments_trace.endswith("npz"):
                    b = np.load(scheduler_dir + "/" + directory + "/" + experiments_trace)
                    record['prediction_overhead'] = b['scheduling_overhead']
                    record['request_latencies'] = b['request_latencies']
                    record['prediction_errors'] = b['sampled_mean_error_ratios']
                    record['compare_error_rate'] = b['sampled_predict_accuracies']
                    record['serving_latencies'] = b['sampled_serving_latencies']
                    record['min_predicted_latency'] = b['sampled_predict_latency']
                    record['sampled_selected_instance_rank'] = b['sampled_selected_instance_rank']

    plot_per_qps(experiments_set, args.output_dir)

    # if args.plot_per_scheduler:
    #     plot_per_scheduler(experiments_set, args.output_dir)


if __name__ == "__main__":
    main()
