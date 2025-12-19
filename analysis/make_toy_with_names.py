#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Create a toy pairs file with demographic names")
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--names_json", type=str, default="/home/zs7353/resume-fting/data/names.json")
    p.add_argument("--limit", type=int, default=100)
    return p.parse_args()


def load_names(path: Path) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        names = json.load(f)
    men_w = names.get("MEN", {}).get("W", [])
    women_w = names.get("WOMEN", {}).get("W", [])
    # fallback guards
    if not isinstance(men_w, list):
        men_w = []
    if not isinstance(women_w, list):
        women_w = []
    return {"WM": men_w, "WW": women_w}


def replace_name(md: str, new_name: str) -> str:
    out_lines: List[str] = []
    replaced = False
    for line in md.splitlines():
        if not replaced and line.strip().lower().startswith("name:"):
            out_lines.append(f"Name: {new_name}")
            replaced = True
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


def main() -> None:
    args = parse_args()
    name_buckets = load_names(Path(args.names_json))
    wm = name_buckets.get("WM", [])
    ww = name_buckets.get("WW", [])
    if not wm or not ww:
        raise RuntimeError("Names not loaded from names.json (WM/WW)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(args.input, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if count >= args.limit:
                break
            try:
                row: Dict[str, Any] = json.loads(line)
            except Exception:
                continue
            base = row.get("base_resume") or ""
            var  = row.get("variant_resume") or ""
            if not base or not var:
                continue
            # Round-robin pick
            i = count % min(len(wm), len(ww))
            name1 = wm[i]
            name2 = ww[i]
            row["base_resume"] = replace_name(base, name1)
            row["variant_resume"] = replace_name(var, name2)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} examples ➜ {out_path}")


if __name__ == "__main__":
    main()


