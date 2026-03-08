#!/bin/bash

# Usage:
#   bash FlexiCache/LongBench.sh <Llama8b|Mistral7b|Mistral24b|Qwen32b>

MODEL="$1"

cd benchmarks/FlexiCache/Language_Modelling/LongBench

./run_longbench.sh "$MODEL" qasper multifieldqa_en 2wikimqa musique gov_report multi_news triviaqa repobench-p