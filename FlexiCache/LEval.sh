#!/bin/bash

MODEL="$1"

cd benchmarks/FlexiCache/Language_Modelling/L-Eval

 ./run_leval.sh "$MODEL"