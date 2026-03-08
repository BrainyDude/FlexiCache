#!/bin/bash

# Usage:
#   bash run_longbench.sh <Llama8b|Mistral7b|Mistral24b|Qwen32b> [task1 task2 ...]
# Example:
#   bash run_longbench.sh Llama8b
#   bash run_longbench.sh Llama8b narrativeqa qasper gov_report
# Pass the model name first; optional task names limit execution to that subset.

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

python eval.py

python make_longbench_table_csv.py --pred_json results/pred.json --out_csv longbench_table.csv