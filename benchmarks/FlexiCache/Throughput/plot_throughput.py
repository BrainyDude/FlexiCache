import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_DIR = Path("Results")
OUT_DIR = Path("plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Map the last filename field to the legend label you want.
# Change these if your naming convention changes.
FC_LABELS = {
    0: "vLLM",
    64: "FlexiCache 1024",
    128: "FlexiCache 2048",
}

# Optional nicer titles / output filenames
MODEL_TITLES = {
    "Llama-3.1-8B-Instruct": "Token Throughput for Llama-3.1-8B",
    "Mistral-7B-Instruct-v0.2": "Token Throughput for Mistral-7B",
}

MODEL_FILENAMES = {
    "Llama-3.1-8B-Instruct": "llama_throughput.png",
    "Mistral-7B-Instruct-v0.2": "mistral_throughput.png",
}

# Filename format:
# FC-{true|false}-{model}-{input_len}-{output_len}-{num_requests}-{fc_key}.json
FILE_RE = re.compile(
    r"^FC-(true|false)-(.+)-(\d+)-(\d+)-(\d+)-(\d+)\.json$"
)


def parse_result_file(path: Path):
    m = FILE_RE.match(path.name)
    if not m:
        return None

    fc_enabled = m.group(1) == "true"
    model = m.group(2)
    input_len = int(m.group(3))
    output_len = int(m.group(4))
    num_requests = int(m.group(5))
    fc_key = int(m.group(6))

    with open(path, "r") as f:
        data = json.load(f)

    tokens_per_second = data["tokens_per_second"]

    return {
        "path": path,
        "fc_enabled": fc_enabled,
        "model": model,
        "input_len": input_len,
        "output_len": output_len,
        "num_requests": num_requests,
        "fc_key": fc_key,
        "tokens_per_second": tokens_per_second,
    }


def load_all_results(results_dir: Path):
    by_model = {}

    for path in sorted(results_dir.glob("*.json")):
        item = parse_result_file(path)
        if item is None:
            print(f"Skipping unmatched filename: {path.name}")
            continue

        model = item["model"]
        fc_key = item["fc_key"]
        out_len = item["output_len"]
        tps = item["tokens_per_second"]

        by_model.setdefault(model, {})
        by_model[model].setdefault(fc_key, {})
        by_model[model][fc_key][out_len] = tps

    return by_model


def plot_model(model: str, series_dict: dict):
    plt.figure(figsize=(8, 5.5))

    # Plot in the order you want in the legend
    plot_order = [0, 64, 128]

    markers = {
        0: "o",
        64: "s",
        128: "x",
    }

    for fc_key in plot_order:
        if fc_key not in series_dict:
            continue

        x_vals = sorted(series_dict[fc_key].keys())
        y_vals = [series_dict[fc_key][x] for x in x_vals]
        label = FC_LABELS.get(fc_key, f"FlexiCache {fc_key}")

        plt.plot(
            x_vals,
            y_vals,
            marker=markers.get(fc_key, "o"),
            linewidth=2.5,
            markersize=8,
            label=label,
        )

    plt.xlabel("Output Lengths", fontsize=18)
    plt.ylabel("Throughput (tokens/s)", fontsize=18)
    plt.title(MODEL_TITLES.get(model, model), fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=14)
    plt.tight_layout()

    out_name = MODEL_FILENAMES.get(
        model,
        model.replace("/", "_").replace(" ", "_") + "_throughput.png",
    )
    out_path = OUT_DIR / out_name
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


def main():
    results = load_all_results(RESULTS_DIR)

    if not results:
        raise RuntimeError(f"No valid JSON result files found in {RESULTS_DIR}")

    for model, series_dict in results.items():
        plot_model(model, series_dict)


if __name__ == "__main__":
    main()