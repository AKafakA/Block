import argparse
import os
import re
from operator import index

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import matplotlib.pyplot as plt
import numpy as np
import shutil
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from experiments_analysis.experiment_plot import plot_linear_for_multiple_qps, directory_name_parser

experiment_name_replacement = {"min latency": "Block"}
scheduler_name_ordered = ['Block']


def directory_name_parser_for_auto_provision(directory_name):
    directory_name = directory_name.split("_")
    qps = directory_name[1]
    n = directory_name[6]
    enable_preemptive_provision = directory_name[23] == "true"
    waiting_time_slo = int(directory_name[18])
    enable_auto_scaling = waiting_time_slo > 0
    return qps, n, enable_preemptive_provision, enable_auto_scaling, waiting_time_slo


def extract_num_available_instances_plot(num_instances_list, min_instances=6):
    """
        generate line outline for the number of available instances
    """
    instance_jump_points = {min_instances: 0}
    for i in range(len(num_instances_list)):
        if num_instances_list[i] not in instance_jump_points:
            instance_jump_points[num_instances_list[i]] = i
    num_instance_curve = []
    current_instance = num_instances_list[0]
    turn_point = [instance_jump_points[num_instance]
                  for num_instance in instance_jump_points.keys() if num_instance > current_instance]
    for i in range(len(num_instances_list)):
        if i in turn_point:
            current_instance += 1
        num_instance_curve.append(current_instance)
    return num_instance_curve


def plot_dual_timeline_data(experiments_set, latency_ax, gpu_free_ax, gpu_variance_ax, instance_ax, font_size=12,
                            provisioning_threshold=60, sigma=10):
    """
    Plot two sets of data on the same timeline with different y-axes.
    """
    initial_text_height = 80
    text_info = {}
    for exp in experiments_set:
        enable_preemptive_provision = exp['enable_preemptive_provision']
        enable_auto_scaling = exp['enable_auto_scaling']
        if enable_preemptive_provision:
            label1 = "Preempt"
            color1 = "blue"
        elif enable_auto_scaling:
            label1 = "Relief"
            color1 = "orange"
        else:
            label1 = "Static"
            color1 = "green"
        latencies = exp['request_latencies']
        num_request_with_latencies_more_than_threshold = np.sum(latencies >= provisioning_threshold)
        available_instances = exp['available_instances']
        avg_gpu_blocks = exp['avg_gpu_blocks']
        var_gpu_blocks = exp['var_gpu_blocks']

        smoothed_avg_gpu_blocks = gaussian_filter1d(avg_gpu_blocks, sigma=sigma)
        smoothed_var_gpu_blocks = gaussian_filter1d(var_gpu_blocks, sigma=sigma)

        x = np.arange(len(latencies))
        # ttft = gaussian_filter1d(ttft, sigma=5)
        # ax1.plot(x, ttft, label=label1, color=color1, linewidth=2)
        latency_ax.scatter(x, latencies, label=label1, color=color1, s=1)
        p99_latency = np.percentile(latencies, 99)
        latency_ax.fill_between(x, latencies, color=color1, alpha=0.1)

        latency_ax.set_ylabel("Latency (s)", fontsize=font_size)
        instance_ax.scatter(x, available_instances, label=label1, color=color1, s=0.01)
        instance_ax.set_ylabel("Available Instances", fontsize=font_size)
        instance_ax.set_xlabel("Query ID", fontsize=font_size)

        num_instance_curve = extract_num_available_instances_plot(available_instances, min_instances=6)
        instance_ax.plot(x, num_instance_curve, label=label1, color=color1, linewidth=0.8, linestyle='--')

        gpu_free_ax.plot(x, smoothed_avg_gpu_blocks, label=label1, color=color1, linewidth=2)
        gpu_free_ax.set_ylabel("Avg GPU Blocks", fontsize=font_size)
        gpu_variance_ax.plot(x, smoothed_var_gpu_blocks, label=label1, color=color1, linewidth=2)
        gpu_variance_ax.set_ylabel("Var GPU Blocks", fontsize=font_size)

        gpu_variance_ax.set_xlabel("Query ID", fontsize=font_size)

        latency_ax.plot([0, len(experiments_set[0]['request_latencies'])], [provisioning_threshold, provisioning_threshold],
                 color='red', linewidth=1)

        latency_ax.text(0, provisioning_threshold + 10, "Scaling SLO", color='red', fontsize=font_size,
                        verticalalignment='bottom')

        latency_ax.legend(loc='upper left', fontsize=font_size, ncol=3, bbox_to_anchor=(0.2, 1.25),
                          scatterpoints=1, markerscale=10)
        gpu_free_ax.legend(loc='upper left', fontsize=font_size, ncol=3, bbox_to_anchor=(0.2, 1.25),
                           fancybox=False, shadow=False)

        text_info[label1] = (num_request_with_latencies_more_than_threshold, p99_latency, color1)

    text_sort = ["Relief", "Preempt", "Static",]
    text_sort.reverse()
    for label in text_sort:
        num_request_with_latencies_more_than_threshold, p99_latency, color = text_info[label]
        latency_ax.text(7000, initial_text_height,
                        f"{label}:{p99_latency:.1f}, {num_request_with_latencies_more_than_threshold}",
                        color=color, fontsize=font_size, verticalalignment='bottom')
        initial_text_height += 10
    latency_ax.text(6000, initial_text_height, "Scheduler:(P99 lat, Num>SLO)", fontsize=font_size, verticalalignment='bottom')


def plot_per_qps(experiments_set, output_dir, selected_qps, provisioning_threshold=60):
    selected_experiments = [experiment for experiment in experiments_set if experiment['qps'] == selected_qps]
    if not selected_experiments:
        print(f"No experiments found for QPS: {selected_qps}")
        return
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2)
    fig.subplots_adjust(hspace=0.2, wspace=0.1)
    fig.set_size_inches(16, 6)
    latency_ax, gpu_free_ax, instance_ax, gpu_varius_ax = axes.flatten()
    plot_dual_timeline_data(selected_experiments, latency_ax, gpu_free_ax, gpu_varius_ax, instance_ax,
                            provisioning_threshold=provisioning_threshold)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/auto_provision.png", dpi=300)


def main():
    parser = argparse.ArgumentParser(description='Plot the results of the experiments')
    parser.add_argument("--experiments-dir", type=str,
                        default="experiments_analysis/auto_provision_experiment_output/sharegpt")
    parser.add_argument("--output-dir", type=str, default="./experiments_analysis/auto_provision_plots")
    parser.add_argument("--plot-per-qps", type=bool, default=True)
    parser.add_argument("--max-instances", type=int, default=12)
    parser.add_argument("--min-instances", type=int, default=6)
    parser.add_argument("--provisioning-threshold", type=int, default=70)
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
            qps, n, enable_preemptive_provision, enable_auto_scaling, waiting_time_slo \
                = directory_name_parser_for_auto_provision(directory)
            record["qps"] = float(qps)
            record["n"] = int(n)
            record["enable_preemptive_provision"] = enable_preemptive_provision
            record["enable_auto_scaling"] = enable_auto_scaling
            record["waiting_time_slo"] = waiting_time_slo
            scheduler_trace = scheduler_dir + "/" + directory + "/running_logs/global_scheduler.log"
            for experiments_trace in os.listdir(scheduler_dir + "/" + directory):
                if experiments_trace.endswith("npz"):
                    b = np.load(scheduler_dir + "/" + directory + "/" + experiments_trace)
                    record['request_latencies'] = b['request_latencies'] / 1000.0  # Convert to seconds

                    record['available_instances'] = b["num_available_instances"]
                    record['avg_gpu_blocks'] = b['avg_gpu_blocks']
                    record['var_gpu_blocks'] = b['var_gpu_blocks']

    for qps in set([experiment['qps'] for experiment in experiments_set]):
        plot_per_qps(experiments_set, args.output_dir, qps,args.provisioning_threshold)

    # if args.plot_per_scheduler:
    #     plot_per_scheduler(experiments_set, args.output_dir)


if __name__ == "__main__":
    main()
