#!/usr/bin/env bash

# Usage:
#   bash Mistral24b.sh
#   bash Mistral24b.sh 100 500 1000
# If no output lengths are provided, uses the default hardcoded list.
# If one or more output lengths are provided, runs only for those lengths.

export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND="TRITON_ATTN_VLLM_V1"

################################## Variables ##################################

NUM_PROMPT=500
INPUT_LEN=30000
DEFAULT_OUTPUT_LENS=(50 100 250 500 750 1000 1250 1500)

if [[ $# -eq 0 ]]; then
  OUTPUT_LENS=("${DEFAULT_OUTPUT_LENS[@]}")
else
  OUTPUT_LENS=("$@")
fi

model="mistralai/Mistral-Small-24B-Instruct-2501"

ENABLE_FLEXICACHE_LIST=(false true)
TOPK_LIST=(0 64)

############################## Running Benchmark ##############################

ratio_in=0.3333
ratio_out=1

dataset_path="Prompts/prompts-Mistral-7B-Instruct-v0.2.json"

OUT_DIR="Results"
mkdir -p "$OUT_DIR"

for OUTPUT_LEN in "${OUTPUT_LENS[@]}"; do

  for i in "${!ENABLE_FLEXICACHE_LIST[@]}"; do

    ENABLE_FLEXICACHE=${ENABLE_FLEXICACHE_LIST[$i]}
    TOP_K=${TOPK_LIST[$i]}
  
    OUT_FILE="${OUT_DIR}/FC-${ENABLE_FLEXICACHE}-${model##*/}-${INPUT_LEN}-${OUTPUT_LEN}-${NUM_PROMPT}-${TOP_K}.json"

    CMD=(
      python "../../benchmark_throughput.py" \
        --dataset-name leval \
        --dataset-path "$dataset_path" \
        --model "$model" \
        --gpu-memory-utilization 0.95 \
        --tensor-parallel-size 1 \
        --no-enable-prefix-caching \
        --disable-cascade-attn \
        --max-num-batched-tokens 32768 \
        --max-num-seqs 64 \
        --input-len "$INPUT_LEN" \
        --output-len "$OUTPUT_LEN" \
        --num-prompts "$NUM_PROMPT" \
        --random-range-ratio-input "$ratio_in" \
        --random-range-ratio-output "$ratio_out" \
        --seed 42 \
        --output-json "$OUT_FILE"
    )

    if [ "$ENABLE_FLEXICACHE" = true ]; then
      CMD+=(
        --rerank-frequency 16 \
        --topK-budget "$TOP_K" \
        --num-unstable-heads 64 \
        --unstable_heads_profile_task gov_report \
        --enable-flexicache
      )
    fi

    # ========================================================================
    # NOTE: HARDCODED NUMA BINDING (node 0)
    # We bind CPU threads and memory allocation to NUMA node 0 for locality.
    # This is optimal only if the selected GPU has NUMA affinity to node 0.
    # Otherwise, remote NUMA memory / cross-socket traffic can reduce throughput.
    #
    # On single-socket (non-NUMA) systems this has no effect.
    #
    # Check GPU locality with `nvidia-smi topo -m` (NUMA Affinity / CPU Affinity)
    # and adjust --cpunodebind/--membind if necessary.
    # ======================================================================== 
    numactl --cpunodebind=0 --membind=0 "${CMD[@]}"

  done
done