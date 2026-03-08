#!/bin/bash

# Usage:
#   bash FlexiCache/Throughput.sh <Llama8b|Mistral7b|Mistral24b|Qwen32b>

MODEL="$1"

cd benchmarks/FlexiCache/Throughput

./run_throughput.sh "$MODEL" 100 500 1000 1500