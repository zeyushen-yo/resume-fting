import json
from collections import Counter
from pathlib import Path


def aggregate(path: str):
    counts = Counter()
    pair_type_counts = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            counts["total"] += 1
            if row.get("valid"):
                counts["valid"] += 1
            else:
                counts["invalid"] += 1
            pt = row.get("pair_type", "?")
            pair_type_counts[pt] += 1
    return counts, pair_type_counts


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=str, nargs="+", default=[
        "/home/zs7353/resume_validity/data/eval_model.jsonl",
        "/home/zs7353/resume_validity/data/eval_agentic.jsonl",
    ])
    args = ap.parse_args()

    for path in args.inputs:
        counts, pt = aggregate(path)
        total = counts["total"] or 1
        print(f"Results for {path}")
        print(f"  Total: {counts['total']}")
        print(f"  Valid: {counts['valid']} ({counts['valid']/total:.2%})")
        print(f"  Invalid: {counts['invalid']} ({counts['invalid']/total:.2%})")
        print(f"  Pair type distribution: {dict(pt)}")


if __name__ == "__main__":
    main()


