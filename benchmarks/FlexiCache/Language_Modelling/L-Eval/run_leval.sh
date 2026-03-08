#!/bin/bash
set -e 

MODEL="$1"

if [[ -z "$MODEL" ]]; then
    echo "Error: No model provided."
    echo "Usage: $0 <Llama8b|Mistral7b|Mistral24b|Qwen32b>"
    exit 1
fi

case "$MODEL" in
    Llama8b|Mistral7b|Mistral24b|Qwen32b)
        bash "./${MODEL}.sh"
        ;;
    *)
        echo "Error: Invalid model '$MODEL'"
        echo "Allowed values: Llama8b Mistral7b Mistral24b Qwen32b"
        exit 1
        ;;
esac

./eval.sh

python make_leval_table_csv.py --results_dir LEval/results --out_csv leval_table.csv