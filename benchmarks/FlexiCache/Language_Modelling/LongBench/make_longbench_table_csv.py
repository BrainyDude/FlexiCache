import argparse
import csv
import json
from typing import Dict, Any, List, Tuple, Optional

TASK_ORDER: List[Tuple[str, str]] = [
    ("narrativeqa", "NarrativeQA"), ("qasper", "Qasper"), ("multifieldqa_en", "MultiField-en"),
    ("hotpotqa", "HotpotQA"), ("2wikimqa", "2WikiMQA"), ("musique", "Musique"),
    ("gov_report", "GovReport"), ("qmsum", "QMSum"), ("multi_news", "MultiNews"),
    ("trec", "TREC"), ("triviaqa", "TriviaQA"), ("samsum", "SAMSum"), ("passage_count", "PCount"),
    ("passage_retrieval_en", "PRe"), ("lcc", "Lcc"), ("repobench-p", "RB-P"),
]

DENSE_KEY = "no_flexicache"

def fmt2(x: Optional[float]) -> str:
    return "" if x is None else f"{x:.2f}"

def pick_flexicache_key(task_dict: Dict[str, Any]) -> Optional[str]:
    """
    Pick the single flexicache variant key for a task.
    If multiple exist, prefer one containing 'flexicache' and '64-unstable-16' if present,
    else the first flexicache-ish key.
    """
    keys = list(task_dict.keys())
    fc = [k for k in keys if k != DENSE_KEY and "flexicache" in k.lower()]
    if not fc:
        # fall back to any non-dense key
        other = [k for k in keys if k != DENSE_KEY]
        return other[0] if other else None

    preferred = [k for k in fc if "64-unstable-16" in k]
    return preferred[0] if preferred else fc[0]

def model_to_flexicache_key(model: str) -> str:
    if "Meta-Llama-3.1-8B" in model or "Mistral-7B" in model:
        return "flexicache-64-unstable-16-rerank-64-topK"
    if "Mistral-Small-24B" in model:
        return "flexicache-80-unstable-16-rerank-64-topK"
    if "Qwen2.5-32B" in model:
        return "flexicache-128-unstable-16-rerank-64-topK"
    assert False, f"Unknown model name: {model}"

def display_model_name(model: str) -> str:
    if "Meta-Llama-3.1-8B" in model:
        return "Llama-3.1-8B"
    if "Mistral-7B" in model:
        return "Mistral-7B-v0.2"
    if "Mistral-Small-24B" in model:
        return "Mistral-Small-24B"
    if "Qwen2.5-32B" in model:
        return "Qwen2.5-32B"
    assert False, f"Unknown model name: {model}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_json")
    ap.add_argument("--out_csv")
    args = ap.parse_args()

    with open(args.pred_json, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    models = sorted(data.keys())
    assert models, "pred.json has no top-level keys (models)."

    model_display = [display_model_name(m) for m in models]

    header = ["Task"]
    for md in model_display:
        header += [f"{md} Dense", f"{md} FlexiCache"]

    rows: List[List[str]] = []
    ratios_per_model: List[List[float]] = [[] for _ in models]

    for task_key, task_name in TASK_ORDER:
        row = [task_name]
        for mi, model in enumerate(models):
            task_block = data.get(model).get(task_key, {})
            dense = task_block.get(DENSE_KEY)
            flex = task_block.get(model_to_flexicache_key(model))

            row += [fmt2(dense), fmt2(flex)]

            if dense is not None and flex is not None:
                ratios_per_model[mi].append(float(flex) / float(dense))

        rows.append(row)

    avg_row = ["Avg. Ratio"]
    for mi in range(len(models)):
        avg = sum(ratios_per_model[mi]) / len(ratios_per_model[mi])
        avg_row += ["-", fmt2(avg)]

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
        w.writerow(avg_row)

    print(f"Wrote: {args.out_csv}")

if __name__ == "__main__":
    main()