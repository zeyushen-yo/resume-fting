#!/usr/bin/env python3
import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser("Merge shard JSONL files into one")
    p.add_argument("--shards_dir", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    shards_dir = Path(args.shards_dir)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fout:
        for p in sorted(shards_dir.glob("pairs_shard_*_of_*.jsonl")):
            with open(p, "r", encoding="utf-8") as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line)
    print(f"Merged into {out}")


if __name__ == "__main__":
    main()


