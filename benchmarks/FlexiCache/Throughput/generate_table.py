import os
import json
import csv
import re
from collections import defaultdict

METRIC_KEY = "tokens_per_second"

RESULTS_DIR = "Results"
OUTPUT_CSV = "throughput_table.csv"

# Expected filename format:
# FC-false-Llama-3.1-8B-Instruct-30000-100-500-0.json
# FC-true-Llama-3.1-8B-Instruct-30000-100-500-64.json
#
# We extract:
#   FC-true / FC-false  -> FlexiCache on/off
#   output length       -> second last numeric group before num_requests and budget
#
# Pattern explanation:
# FC-(true|false)-...-(context_len)-(output_len)-(num_requests)-(budget).json
pattern = re.compile(
    r"^FC-(true|false)-.+-(\d+)-(\d+)-(\d+)-(\d+)\.json$"
)

results = defaultdict(dict)

for fname in os.listdir(RESULTS_DIR):
    if not fname.endswith(".json"):
        continue

    m = pattern.match(fname)
    if not m:
        print(f"Skipping unmatched file: {fname}")
        continue

    fc_enabled = m.group(1) == "true"
    context_len = int(m.group(2))   # unused, but parsed for completeness
    output_len = int(m.group(3))
    num_requests = int(m.group(4))  # unused
    budget = int(m.group(5))        # unused

    fpath = os.path.join(RESULTS_DIR, fname)
    with open(fpath, "r") as f:
        data = json.load(f)

    if METRIC_KEY not in data:
        raise KeyError(f"{METRIC_KEY} not found in {fname}")

    value = float(data[METRIC_KEY])

    if fc_enabled:
        results[output_len]["FlexiCache-1024"] = value
    else:
        results[output_len]["vLLM"] = value

# Write CSV
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Output Length", "vLLM", "FlexiCache-1024", "Speedup"])

    for output_len in sorted(results):
        vllm = results[output_len].get("vLLM")
        flexi = results[output_len].get("FlexiCache-1024")

        if vllm is None or flexi is None:
            print(f"Skipping output length {output_len}: missing pair")
            continue

        speedup = flexi / vllm if vllm != 0 else 0.0

        writer.writerow([
            output_len,
            f"{vllm:.1f}",
            f"{flexi:.1f}",
            f"{speedup:.2f}x",
        ])

print(f"Saved CSV to: {OUTPUT_CSV}")