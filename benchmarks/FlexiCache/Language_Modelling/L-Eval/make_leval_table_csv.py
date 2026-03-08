import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

TASK_DISPLAY = {
    "financial_qa": "LongFQA", "gov_report_summ": "GovReport", "legal_contract_qa": "CUAD",
    "meeting_summ": "QMSum", "news_summ": "Multi-News", "paper_assistant": "Openreview",
    "patent_summ": "BigPatent", "review_summ": "SPACE", "tv_show_summ": "SummScreen",
}

DISPLAY_TASKS = [
    "LongFQA", "GovReport", "CUAD", "QMSum", "Multi-News", "Openreview", "BigPatent", "SPACE", "SummScreen",
]

ROW_ORDER = [
    "Dense", "FlexiCache W/O Reranking", "FlexiCache W/O Unstable Heads", "FlexiCache 1024", "FlexiCache 2048",
]

def model_method_from_filename(fname: str) -> tuple[str, str]:
    base = os.path.basename(fname)

    if base.endswith("-no-flexicache.jsonl"):
        model = base[: -len("-no-flexicache.jsonl")]
        return model, "Dense"

    m = re.match(
        r"^(?P<model>.+)-(?P<unstable>\d+)-unstable-(?P<rerank>\d+)-rerank-topK-(?P<topk>\d+)\.jsonl$",
        base,
    )
    if not m:
        raise SystemExit(f"Filename doesn't match expected patterns: {fname}")

    model = m.group("model")
    unstable = int(m.group("unstable"))
    rerank = int(m.group("rerank"))
    topk = int(m.group("topk"))

    if unstable == 0 and rerank >= 10000 and topk == 64:
        return model, "FlexiCache W/O Reranking"
    if unstable == 0 and rerank == 16 and topk == 64:
        return model, "FlexiCache W/O Unstable Heads"
    if unstable in (64,80,128) and rerank == 16 and topk == 64:
        return model, "FlexiCache 1024"   # 64 pages * 16 tokens/page = 1024 tokens
    if unstable in (64,80,128) and rerank == 16 and topk == 128:
        return model, "FlexiCache 2048"   # 128 pages * 16 tokens/page = 2048 tokens

    raise SystemExit(f"Filename doesn't match expected patterns: {fname}")

def read_scores_jsonl(path: str) -> dict[str, float]:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            task = obj["task"]
            score = float(obj["LEval_score"])
            out[task] = score
    return out

def fmt2(x: float | None) -> str:
    if x is None:
        return ""
    return f"{x:.2f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir")
    ap.add_argument("--out_csv")
    args = ap.parse_args()

    # scores[model][method][display_task] = score
    scores = defaultdict(lambda: defaultdict(dict))

    files = sorted(glob.glob(os.path.join(args.results_dir, "*.jsonl")))
    assert files, f"No .jsonl files found under: {args.results_dir}"

    for fp in files:
        model, method = model_method_from_filename(fp)
        raw = read_scores_jsonl(fp)

        for task_key, display in TASK_DISPLAY.items():
            if task_key in raw:
                scores[model][method][display] = raw[task_key]

    # Write CSV: one header row, then a model "section" row, then method rows.
    header = ["Task"] + DISPLAY_TASKS + ["Avg. Ratio"]

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        for model in sorted(scores.keys()):
            # model section row
            w.writerow([model] + [""] * (len(DISPLAY_TASKS) + 1))

            dense = scores[model].get("Dense")

            for method in ROW_ORDER:
                row_scores = scores[model][method]
                row = [method]

                for col in DISPLAY_TASKS:
                    row.append(fmt2(row_scores.get(col)))

                # Avg. Ratio (mean over tasks of method_score / dense_score)
                if method == "Dense":
                    row.append("")
                else:
                    ratios = []
                    for col in DISPLAY_TASKS:
                        d = dense.get(col)
                        s = row_scores.get(col)
                        if d is None or s is None or d == 0:
                            continue
                        ratios.append(s / d)
                    avg_ratio = (sum(ratios) / len(ratios)) if ratios else None
                    row.append(fmt2(avg_ratio))

                w.writerow(row)

            w.writerow([]) # blank line between models

    print(f"Wrote: {args.out_csv}")

if __name__ == "__main__":
    main()