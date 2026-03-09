# FlexiCache Artifact Evaluation

**FlexiCache: Leveraging Temporal Stability of Attention Heads for Efficient KV Cache Management**

The artifact evaluation for **MLSys 2026** primarily focuses on **two main results from the paper**:

1. **Accuracy retention** on LongBench (~2 hours)
2. **Throughput improvement** over the vLLM baseline (~3 hours)

We first describe the workflow for reproducing the main results for artifact evaluation. Later, we describe in detail how to reproduce additional extensive results presented in the main paper.

The official MLSys artifact appendix is available here: [Artifact_Evaluation.pdf](Artifact_Evaluation.pdf)

---

# Overview

FlexiCache is a **stability-aware hierarchical KV-cache management system** that reduces GPU memory pressure and improves long-context LLM inference efficiency.

This artifact allows reviewers to:

- Build FlexiCache (implemented on top of **vLLM**)
- Run automated scripts reproducing the main experiments
- Generate CSV files containing the results

---

# Hardware Requirements

Experiments were tested on:

- 1× NVIDIA H100 GPU (94GB recommended, 80GB minimum)
- ≥256GB host RAM
- PCIe Gen5 CPU–GPU interconnect
- Ability to allocate ~180GB pinned host memory for KV cache offloading

For reliable throughput measurements, the machine should be idle with no other workloads competing for GPU/CPU/RAM bandwidth.

---

# Software Requirements

- Linux (we used Ubuntu 24.04.2 LTS)
- Anaconda (we used 24.9.2)
- Python 3.12
- PyTorch 2.6.0+cu124
- Triton 3.2.0
- Transformers 4.50.0
- Datasets 3.6.0
- CUDA NVIDIA driver (we tested on CUDA 12.8)

---

# Artifact Checklist

| Item                                                             | Description                                                                                                                                                                              |
| ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Algorithm                                                        | Stability-aware hierarchical KV-cache management and sparse decode attention                                                                                                             |
| Program                                                          | LongBench evaluation scripts and throughput benchmarking scripts for FlexiCache and vLLM                                                                                                 |
| Compilation                                                      | Python package build via `pip install -e .`, compiling custom CUDA, Triton, and C++ extensions for FlexiCache on top of vLLM                                                             |
| Model                                                            | `meta-llama/Llama-3.1-8B-Instruct`, downloaded automatically from Hugging Face when running the scripts; requires Internet access and a configured Hugging Face account                  |
| Data set                                                         | `zai-org/LongBench`, downloaded automatically from Hugging Face when running the scripts; requires Internet access                                                                       |
| Run-time environment                                             | Linux, Conda, Python 3.12, PyTorch 2.6.0+cu124, Triton 3.2.0, Transformers 4.50.0, CUDA compatible NVIDIA driver                                                                         |
| Hardware                                                         | An x86_64 Linux machine with an NVIDIA H100 GPU (94GB recommended, 80GB minimum), at least 256GB of host memory and PCIe Gen5 CPU-GPU interconnect.                                      |
| Run-time state                                                   | For reliable throughput measurements, experiments should be run on an idle machine without other workloads contending for the GPU, CPU, RAM, or host–device bandwidth                    |
| Execution                                                        | Single-GPU execution via automated bash scripts. The artifact uses a large pinned host-memory pool (about 180GB by default) and frequent host–device KV-cache transfers during execution |
| Metrics                                                          | LLM generation throughput (tokens/second). Long-context benchmarks accuracy                                                                                                              |
| Output                                                           | Accuracy numbers and generation throughput presented in 2 CSV files                                                                                                                      |
| Experiments                                                      | (1) LongBench accuracy retention compared to dense attention baseline (2) end-to-end throughput gain compared to vLLM baseline                                                           |
| How much disk space required (approximately)?                    | 100GB                                                                                                                                                                                    |
| How much time is needed to prepare workflow (approximately)?     | Around 1 hour to build FlexiCache (implemented on top of vLLM) from source                                                                                                               |
| How much time is needed to complete experiments (approximately)? | Around 2 hours for the accuracy retention experiment. Around 3 hours for the throughput experiment                                                                                       |
| Publicly available?                                              | Yes                                                                                                                                                                                      |
| Code licenses (if publicly available)?                           | Apache License 2.0                                                                                                                                                                       |
| Data licenses (if publicly available)?                           | MIT                                                                                                                                                                                      |
| Workflow framework used?                                         | Bash scripts and Python evaluation code                                                                                                                                                  |
| Archived (provide DOI)?                                          | https://doi.org/10.5281/zenodo.18910205                                                                                                                                                  |

---

## Installation

We recommend using Anaconda to isolate the software dependencies. Installation takes approximately **1 hour**, primarily due to building vLLM from source.

```bash
# 1. Clone the repository
git clone https://github.com/NazmulTakbir/FlexiCache.git
cd FlexiCache

# 2. Create and activate a Conda environment
conda create -n FlexiCache python=3.12 -y
conda activate FlexiCache

# 3. Set CUDA environment variables so nvcc is found during build (paths may vary by machine)
export CUDA_HOME=/usr/local/cuda-12.8
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
export PATH=/usr/local/cuda-12.8/bin:$PATH

# 4. Build from source
export MAX_JOBS=10
pip install -e .  # ~30 min to 1 hour

# 5. Install additional dependencies
pip install -r flexicache_requirements.txt

# 6. Authenticate with Hugging Face (Required for Llama 3.1)
# You must also accept the model agreement on Hugging Face
pip install huggingface_hub
huggingface-cli login
```

## Experiment workflow

We provide two push-button bash scripts to reproduce the primary accuracy and throughput results for `meta-llama/Llama-3.1-8B-Instruct`. Both commands should be run from the repository root after installation.

### Accuracy on LongBench

This script evaluates FlexiCache on a subset of **8 LongBench tasks**.

```bash
# Run
bash FlexiCache/LongBench.sh Llama8b

# Results are written to:
benchmarks/FlexiCache/Language_Modelling/LongBench/longbench_table.csv
```

This workflow takes approximately 2 hours.

### Throughput

This script benchmarks end-to-end throughput (tokens/sec) of **FlexiCache** (with a 1024-token budget) against the baseline **vLLM** for output lengths of **100, 500, 1000, and 1500 tokens**, using input prompts randomly sampled between **10k and 30k tokens**.

```bash
# Run
bash FlexiCache/Throughput.sh Llama8b

# Results are written to:
benchmarks/FlexiCache/Throughput/throughput_table.csv
```

This workflow takes approximately 3 hours.

## Evaluation and Expected Result

In this section, we provide reference results for comparison. Due to differences in machine environments, absolute throughput values may vary. However, the relative speedup of FlexiCache over the vLLM baseline should remain consistent. Accuracy results may also show small deviations from the reported numbers due to the inherent stochasticity of LLM generation and minor floating-point differences across hardware.

### Accuracy on LongBench

| Task           | Dense | FlexiCache |
| -------------- | ----- | ---------- |
| Qasper         | 45.23 | 45.92      |
| MultiField-en  | 54.93 | 54.94      |
| 2WikiMQA       | 45.45 | 45.32      |
| Musique        | 30.18 | 31.22      |
| GovReport      | 34.98 | 34.22      |
| MultiNews      | 27.18 | 27.10      |
| TriviaQA       | 91.64 | 91.49      |
| RB-P           | 49.43 | 49.70      |
| **Avg. Ratio** | --    | **1.00**   |

**Tab: Accuracy Retention on LongBench**

---

### Throughput (tokens/second)

| Output Length | vLLM    | FlexiCache-1024 | Speedup |
| ------------- | ------- | --------------- | ------- |
| 100           | 11178.7 | 12433.1         | 1.11×   |
| 500           | 7742.1  | 11480.2         | 1.48×   |
| 1000          | 5609.9  | 9965.4          | 1.78×   |
| 1500          | 4549.0  | 9444.9          | 2.08×   |

**Tab: Throughput Gain**

---

# Additional Experiments

For artifact evaluation, we focus on a representative subset of experiments that highlight the two key properties of **FlexiCache**: **accuracy retention** and **throughput improvement**. These experiments use the model `meta-llama/Llama-3.1-8B-Instruct` and a subset of LongBench tasks, and can be completed within a few hours.

If reviewers would like to reproduce more extensive results from the paper, the following sections describe how to run additional experiments.

---

## Accuracy on Full LongBench

By default, `FlexiCache/LongBench.sh` evaluates a subset of **8 LongBench tasks**.

To run evaluation on the **full LongBench benchmark**, simply remove the task list from the command.

Change:

```bash
./run_longbench.sh "$MODEL" qasper multifieldqa_en 2wikimqa musique gov_report multi_news triviaqa repobench-p
```

to:

```bash
./run_longbench.sh "$MODEL"
```

### Accuracy on L-Eval

L-Eval experiments can be executed directly using:

```bash
bash FlexiCache/LEval.sh Llama8b
```

### Throughput with Different KV Cache Budgets

By default, throughput experiments run with a 1024-token KV cache budget (64 pages × 16 tokens per page).
The paper also reports results with a 2048-token budget.

To reproduce those results:

1. Open the file:
   `benchmarks/FlexiCache/Throughput/Llama8b.sh`

2. Change:
   ```bash
   TOPK_LIST=64
   ```
   to:
   ```bash
   TOPK_LIST=128
   ```
3. Run the throughput experiment again:
   ```bash
   bash FlexiCache/Throughput.sh Llama8b
   ```

### Throughput with Different Output Lengths

The default throughput script evaluates output lengths: `100, 500, 1000, 1500`

In the paper, we also report results for a broader range of output lengths: `50, 100, 250, 500, 750, 1000, 1250, 1500`

To reproduce those results:

1. Open the file:
   `FlexiCache/Throughput.sh`

2. Change:
   ```bash
   ./run_throughput.sh "$MODEL" 100 500 1000 1500
   ```
   to:
   ```bash
   ./run_throughput.sh "$MODEL" 50 100 250 500 750 1000 1250 1500
   ```
3. Run the experiment again:
   ```bash
   bash FlexiCache/Throughput.sh Llama8b
   ```

### Running Experiments on Other Models

The same scripts can also be used with other supported models by passing the model identifier as an argument.

Examples are shown below.

```bash
# Mistral-7B-Instruct-v0.2
bash FlexiCache/LongBench.sh Mistral7b
bash FlexiCache/Throughput.sh Mistral7b
bash FlexiCache/LEval.sh Mistral7b
```

```bash
# Mistral-Small-24B-Instruct-2501
bash FlexiCache/LongBench.sh Mistral24b
bash FlexiCache/Throughput.sh Mistral24b
bash FlexiCache/LEval.sh Mistral24b
```

```bash
# Qwen2.5-32B-Instruct
bash FlexiCache/LongBench.sh Qwen32b
bash FlexiCache/Throughput.sh Qwen32b
bash FlexiCache/LEval.sh Qwen32b
```

These commands will automatically download the corresponding models from Hugging Face if they are not already present.
