# Block

**Block: Balancing Load in LLM Serving with Context, Knowledge and Predictive Scheduling** ([arXiv:2508.03611](https://arxiv.org/abs/2508.03611))

> 🎉 **Accepted at SYSTOR 2026.** The conference version appears under the title *"Astrolabe: Balancing Load in LLM Serving with Randomized Prediction-Guided Scheduling."*
>
> **Block** (this repository and the arXiv version above) and **Astrolabe** (the SYSTOR paper) are the same system — the conference version uses the Astrolabe name, and figures produced by the artifact are labelled accordingly.
> <!-- ACM DL link to be added once the DOI is issued: [ACM Digital Library](https://doi.org/TODO) -->
>
> 📦 **Paper artifact:** the exact code, experiment scripts, and figure pipeline used for the SYSTOR '26 paper live on the [`systor-ae`](../../tree/systor-ae) branch — see its [`AE.md`](../../blob/systor-ae/AE.md) for a figure-by-figure/table-by-table reproduction guide.

Block is a research prototype that explores *predictive, performance-aware scheduling* for distributed large-language-model (LLM) inference.
It builds on top of Microsoft’s [Vidur](https://github.com/microsoft/vidur) simulator, which initially developed for offline evaluation and optimal configuration searching and adds

* a side-car *Predictor* service that forecasts per–instance leading metrics with Vidur at run time,
* a *Global Scheduler* whose default policy is **randomized power-of-two-choices (Po2) predictive dispatch**: for each request it samples two random instances, queries their Predictors in parallel, and routes to the lower-predicted-latency one (sampling `k` is tunable; `k = N` recovers full fanout), and
* tooling for training a light-weight length-estimator model so the scheduler can reason about prompts it has never seen before.

The randomized Po2 policy keeps the per-request probing cost constant, cuts per-predictor CPU by ~2.8× vs full fanout, and breaks *scheduler herding* under bursty arrivals — while matching full-fanout and load-aware-baseline serving capacity (see the SYSTOR '26 paper).

Everything needed to reproduce the paper's results — source code, datasets, experiment scripts — lives in this repository; the camera-ready artifact snapshot is on the [`systor-ae`](../../tree/systor-ae) branch.

---

## 1. Architecture at a Glance

![Block Architecture](images/block_structure.png)

• **Predictor** (`block/predictor`)
 Co-locates with every inference node. Collects live stats, or spins up a Vidur simulation on-demand, and answers *“What if I got one more request?”*

• **Global Scheduler** (`block/global_scheduler`)
 Receives requests and applies the scheduling policy. Default: **Block's randomized Po2 predictive dispatch** — sample k=2 instances (`--num_query_predictor`), probe their Predictors, and route to the minimum predicted latency; k=N gives deterministic full fanout. Alternatives: Llumnix-style heuristic, INFaaS++, round-robin, random, min-QPM.

• **Query Length Tagger** (`block/length_estimation`)
 A RoBERTa-based regressor that predicts the response-token count for unseen (model, prompt) pairs, feeding the scheduler with input-aware cost estimates. Currently, we just run this model offline to tag the ShareGPT dataset with predicted response length in `data/trace_data/sharegpt/generate` but it should be easy to adapt to any runtime model service as tf-serving or TorchServe.

Block is inference-engine agnostic. We provide an implementation for vLLM 0.7.2 (see the sealed repo at https://github.com/AKafakA/vllm/tree/block).

---

## 2. Repository Layout

```
block/
 ├── predictor/             # side-car prediction service
 ├── global_scheduler/      # request router
 ├── length_estimation/     # training & inference of token-length regressor
 ├── benchmark/             # Poisson load-generator (forked from vLLM)
 ├── config/
 │     └── host_configs.json# cluster description template
 ├── exp/
 │     ├── generate_config.py
 │     ├── setup.sh         # installs deps & deploys cluster
 │     ├── end_to_end_exp_scripts/
 │     └── …                # run, plot, gather
 └── data/                  # ShareGPT, BurstGPT, ArXiv-Summ datasets

vidur # Same as original vidur repo but with new replica scheduler/revised simulator
 ├──  ...
 ├── scheduler
 	 ├── sarathi_replica_scheduler.py # vLLM scheduler simulator with chunked prefill
	 ├── simulate_predict_replica_scheduler.py # class to adapt other simulators/latency linear model into Predictor Service
	 └── …                # other simulators
```

---

## 3. Quick Start

1. Set up cluster
2. Generate the cluster hosts configuration
   
   If using cloudlab, just download its manifest.xml and moved to block/prediction
   
   ```
   python block/exp/generate_config.py --username USERNAME_SSH_TO_HOST
   ```
   
   Otherwise, need to manually generate configration and hostname listing files as examples under block/config.
3. Deploy software stack and vLLM build
   
   ```bash
   sh block/exp/setup.sh
   ```
   
   Insert the vLLM github link if missing or manually distributed vLLM repo to testing workers
4. Start a full end-to-end experiment (≈50 h on 12 × A30 GPUs)

   Fill your Hugging Face token in `exp/run_exp_vllm.sh` and the hostname to run the global scheduler and benchmarking at `exp/experiment.sh`.
   
   And run all end-to-end scripts like
   
   ```bash
   sh block/exp/end_to_end_exp_scripts/main_experiments.sh
   ...
   # results under experiment_output/data/
   ```
 All testing scripts are located in the `block/exp/end_to_end_exp_scripts` directory:

 **Main experiments (`a30_main/`):**
 - **`main_experiment.sh`**: QPS sweep (20-36), capacity curves — **Figure 6** (§5.3)

 **Ablation studies (`ablation/`):**
 - **`po2_ablation_exp.sh`**: Power-of-Two (N=2) vs N=12 — **§6.8**
 - **`burstiness_exp.sh`**: Gamma arrivals (α=0.25-2.0) — **§6.9**
 - **`error_heatmap_exp.sh`**: Prediction noise sensitivity (4×4 grid) — **§6.10**
 - **`cpu_tracking_experiment.sh`**: Predictor CPU/memory overhead — **§6.11**
 - **`auto_provision_exp.sh`**: Auto-provisioning demonstration — **§5.5** / **Figure 8**
 - **`config_search_experiment.sh`**: Batch size × chunk size sweep — **Table 2** (§5.6)
 - **`extension_experiment.sh`**: Qwen2-7B + BurstGPT — **Table 2** (§5.6)
 - **`prediction_experiment.sh`**: Length prediction accuracy — **Figure 5** (§5.2.2)
 - **`block_nosim_ablation_exp.sh`**: Block-NoSim (no Vidur simulation) ablation

 **A100 supplementary (`a100_supplementary/`):**
 - **`full_comparison.sh`**: Block vs Llumnix (scheduling-only + migration) on A100 — **§6.7**
 - **`a100_llama70b_exp.sh`**: A100 Llama-70B latency sweep
 - **`a100_40gb_profiling.sh`**: Fresh A100-40GB hardware profiling
   
	
6. Plot and summarise results after all above experiments finished 
   
   ```bash
   sh block/exp/end_to_end_exp_scripts/plot.sh
   # figures end up in experiment_output/results/
   ```
---

## 4. Train / Evaluate the Length Estimator

If you wish to regenerate the regressor instead of using the provided tagged data.

Download the dataset from

```
wget https://huggingface.co/datasets/shibing624/sharegpt_gpt4/blob/main/sharegpt_gpt4.jsonl
```

And train and generate the tagged data

```bash
python block/length_estimation/sample 
python block/length_estimation/train_roberta
# Tag ShareGPT prompts with ground-truth response lengths 
python block/length_estimation/eval_roberta --tag-data True
```

---

## 5. Benchmarking with Other dataset

`block/benchmark_serving`, which modified from vLLM benchmark scripts can replay JSON/CSV dataset at a configurable Poisson arrival rate

```bash
python block/benchmark/benchmark_serving.py
```

---

## 6. Extending Block to new Scheduler and Model

• Add a new scheduling heuristic: 1) implement from inference side to export required metrics to Predictor (checking vLLM latest commits), and 2) define its load scores as inside `simulate_predictor.py. predict` 3) Append the name of new scheduler under static enum class `vidur/types/optimal_global_scheduler_target_metric.py` and update inside e2e shell scripts

• Support a different inference engine: expose the metrics in a new API, taking the `scheduler_trace` in vLLM `vllm/entrypoints/api_server.py` at the vLLM implementation as an examples.

• Support different models with different GPU SKUs 
 1. please following the [Vidur instruction](https://github.com/microsoft/vidur/blob/main/docs/profiling.md) to gather the profiling data and moved to `data/profiling`
 2. Append the new model configurations at `vidur/config/model_config.py`. Currently we only profiled the cloudlab d7525 host (with single A30 GPUs) with Qwen2-7B and Llama-2-7B-hf for prototyping.
 3. Provide a new model config for global scheduler to use, following the `block/config/llama_config.json` as an example and update the end-to-end scripts to use the new model config according, checking '/block/end_to_end_exp_scripts/entension' which run the experiments on Llama2 and Qwen2 both.
 4. Finally, if using a estimated length associated with the new model ( as Block* in the paper), following above steps to train/evaluate the length estimator to get the tagged data and put it under `data/trace_data/DATASET_NAME/generate/MODEL_NAME/`

---

## 7. Detailed Documentation

See `docs/` for comprehensive guides:

- **[docs/INDEX.md](docs/INDEX.md)** — Documentation overview and navigation
- **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** — Step-by-step A30 + A100 deployment (installation order, CloudLab setup)
- **[docs/EXPERIMENT_GUIDE.md](docs/EXPERIMENT_GUIDE.md)** — All experiment scripts, parameters, expected runtimes
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — System architecture and data flow
- **[docs/RESULTS_SUMMARY.md](docs/RESULTS_SUMMARY.md)** — Key numbers: capacity, latency, robustness
- **[docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md)** — Critical fixes and operational lessons

---

## 8. Requirements

It was tested with this set of packages

• Python 3.10

• CUDA 12.6

• flashinfer-python 0.2.5 and triton 3.2.0, PyTorch-2.5+, customized vLLM based on 0.7.2

Plase checking requirments.txt and `block/exp/setup.sh`

---

## 9. Citation

If you find Block useful, please cite the SYSTOR '26 paper (BibTeX will be updated once the ACM proceedings entry is live):

```
Wei Da and Evangelia Kalyvianaki. Astrolabe: Balancing Load in LLM Serving with
Randomized Prediction-Guided Scheduling. In Proceedings of the ACM International
Systems and Storage Conference (SYSTOR '26), September 2026.
```

or the extended arXiv version:

```
@misc{da2025blockbalancingloadllm,
      title={Block: Balancing Load in LLM Serving with Context, Knowledge and Predictive Scheduling}, 
      author={Wei Da and Evangelia Kalyvianaki},
      year={2025},
      eprint={2508.03611},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2508.03611}, 
}
```

---

## 10. License

This work is released under the MIT license. See `LICENSE` for details.

Happy scheduling!
