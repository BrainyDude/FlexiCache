#!/bin/bash

# Usage:
#   bash Llama8b.sh
#   bash Llama8b.sh narrativeqa qasper gov_report
# If no task names are provided, runs all LongBench tasks.
# If task names are provided, only those valid tasks are run.

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
export VLLM_USE_V1=1
# export CUDA_VISIBLE_DEVICES=0

MODEL="Meta-Llama-3.1-8B-Instruct"

BATCH_SIZE=15

ALL_DATASETS="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique qmsum gov_report multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p"

if [[ $# -eq 0 ]]; then
    DATASETS="$ALL_DATASETS"
else
    for ds in "$@"; do
        if [[ ! " $ALL_DATASETS " =~ [[:space:]]$ds[[:space:]] ]]; then
            echo "Error: Invalid dataset '$ds'"
            echo "Valid datasets: $ALL_DATASETS"
            exit 1
        fi
    done
    DATASETS="$*"
fi

python run_benchmark.py \
    --model $MODEL --dataset $DATASETS --batch_size $BATCH_SIZE

DATASETS_WO_GOV_REPORT=$(echo "$DATASETS" | sed -e 's/\bgov_report\b//g' | xargs)
python run_benchmark.py \
    --model $MODEL --dataset $DATASETS_WO_GOV_REPORT --batch_size $BATCH_SIZE --flexicache \
    --num_unstable_heads 64 --rerank_frequency 16 --topK_budget 64 --unstable_heads_profile_task gov_report

if [[ " $DATASETS " == *" gov_report "* ]]; then
    python run_benchmark.py \
        --model $MODEL --dataset gov_report --batch_size $BATCH_SIZE --flexicache \
        --num_unstable_heads 64 --rerank_frequency 16 --topK_budget 64 \
        --unstable_heads_profile_task paper_assistant
fi