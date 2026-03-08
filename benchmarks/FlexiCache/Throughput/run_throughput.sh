#!/bin/bash

# Usage:
#   bash run_throughput.sh <Llama8b|Mistral7b|Mistral24b|Qwen32b> [output_len1 output_len2 ...]
# Example:
#   bash run_throughput.sh Llama8b
#   bash run_throughput.sh Llama8b 100 500 1000
# The first argument is the model name; any remaining arguments are forwarded as output lengths.

set -e 

MODEL="$1"

if [[ -z "$MODEL" ]]; then
    echo "Error: No model provided."
    echo "Usage: $0 <Llama8b|Mistral7b|Mistral24b|Qwen32b>"
    exit 1
fi

case "$MODEL" in
    Llama8b|Mistral7b|Mistral24b|Qwen32b)
        bash "./${MODEL}.sh" "${@:2}"
        ;;
    *)
        echo "Error: Invalid model '$MODEL'"
        echo "Allowed values: Llama8b Mistral7b Mistral24b Qwen32b"
        exit 1
        ;;
esac

python plot_throughput.py

python generate_table.py