#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Aggregate real-world evaluation results into core metrics")
    p.add_argument("--eval_dirs", nargs="+", default=[
        "/home/zs7353/resume_validity/evaluations/baseline",
        "/home/zs7353/resume_validity/evaluations/baseline_noabstain",
    ])
    p.add_argument("--output", type=str, default="/home/zs7353/resume_validity/analysis/real_world_metrics.csv")
    p.add_argument("--pairs_root", type=str, default="/home/zs7353/resume_validity/data/pairs_from_real_world")
    return p.parse_args()


def _infer_framework_from_path(fp: Path) -> str:
    s = str(fp).lower()
    if "baseline_noabstain" in s:
        return "baseline_noabstain"
    if "baseline" in s:
        return "baseline"
    return "unknown"


def _infer_model_from_filename(fp: Path) -> str:
    name = fp.name
    m = re.match(r"([^_]+)_paired_resume_decisions_", name)
    if m:
        return m.group(1)
    # fallback: strip extension
    return name.rsplit(".", 1)[0]


def _extract_job_key(row: pd.Series) -> Tuple[str, str]:
    # Prefer job_source if present and parseable; fallback to job_title only
    js = row.get("job_source", "")
    if isinstance(js, str) and js.strip().startswith("{"):
        try:
            d = ast.literal_eval(js)
            company = str(d.get("company") or "").strip()
            title = str(d.get("title") or "").strip()
            if company or title:
                return company, title
        except Exception:
            pass
    # fallback
    return "", str(row.get("job_title") or "").strip()


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    csv_paths: List[Path] = []
    for base in args.eval_dirs:
        p = Path(base)
        if p.is_dir():
            csv_paths.extend(sorted(p.glob("*.csv")))
        elif p.is_file() and p.suffix.lower() == ".csv":
            csv_paths.append(p)
    if not csv_paths:
        print("No evaluation CSVs found.", file=sys.stderr)
        sys.exit(1)

    frames: List[pd.DataFrame] = []
    for fp in csv_paths:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"[WARN] failed to read {fp}: {e}")
            continue
        if df.empty:
            continue
        df["framework"] = _infer_framework_from_path(fp)
        df["model_id"] = _infer_model_from_filename(fp)
        # Normalize
        df["experiment_type"] = df.get("experiment_type_norm", df.get("experiment_type", "")).astype(str).str.lower()
        df["decision"] = df.get("decision", "").astype(str).str.lower()
        df["abstained"] = df.get("abstained", False).astype(bool)
        df["is_valid"] = df.get("is_valid", False).astype(bool)
        # Extract job key
        companies: List[str] = []
        titles: List[str] = []
        for _, r in df.iterrows():
            c, t = _extract_job_key(r)
            companies.append(c)
            titles.append(t)
        df["job_company"] = companies
        df["job_title_posting"] = titles
        frames.append(df)

    if not frames:
        print("No rows loaded from evaluation CSVs.", file=sys.stderr)
        sys.exit(1)

    all_df = pd.concat(frames, ignore_index=True)

    # Filter to only rows corresponding to the real-world pairs under pairs_root
    try:
        import json
        allowed: set[tuple[str, str]] = set()
        for pf in Path(args.pairs_root).rglob("pairs_shard_00_of_01.jsonl"):
            with open(pf, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    js = obj.get("job_source") or {}
                    comp = str(js.get("company") or "").strip()
                    title = str(js.get("title") or "").strip()
                    if comp or title:
                        allowed.add((comp, title))
                    break
        if allowed:
            mask = []
            for _, r in all_df.iterrows():
                c, t = _extract_job_key(r)
                mask.append((c, t) in allowed)
            all_df = all_df[pd.Series(mask)].copy()
    except Exception as e:
        print(f"[WARN] filtering to pairs_root failed: {e}")

    # Metric computation
    results: List[Dict[str, Any]] = []
    group_cols = ["job_company", "job_title_posting", "model_id"]
    for key, g in all_df.groupby(group_cols, dropna=False):
        job_company, job_title_posting, model_id = key
        # Criterion Validity & Unjustified Abstention (strict pairs = num_differed > 0, from baseline)
        g_baseline = g[g["framework"] == "baseline"].copy()
        g_strict = g_baseline[g_baseline["num_differed"].fillna(0) > 0].copy()
        n_strict = len(g_strict)
        cv = float(g_strict["is_valid"].sum()) / n_strict if n_strict > 0 else np.nan
        uja = float(g_strict["abstained"].sum()) / n_strict if n_strict > 0 else np.nan

        # Discriminant Validity (equal pairs = num_differed == 0, abstain rate, from baseline)
        g_equal = g_baseline[g_baseline["num_differed"].fillna(0) == 0].copy()
        n_equal = len(g_equal)
        dv = float(g_equal["abstained"].sum()) / n_equal if n_equal > 0 else np.nan

        # Selection rate for first (forced choice, equal pairs, from baseline_noabstain)
        g_noab = g[g["framework"] == "baseline_noabstain"].copy()
        g_equal_noab = g_noab[g_noab["num_differed"].fillna(0) == 0].copy()
        n_equal_noab = len(g_equal_noab)
        sr_first = float((g_equal_noab["decision"] == "first").sum()) / n_equal_noab if n_equal_noab > 0 else np.nan

        results.append({
            "job_company": job_company,
            "job_title_posting": job_title_posting,
            "model_id": model_id,
            "criterion_validity": cv,
            "unjustified_abstention": uja,
            "discriminant_validity": dv,
            "selection_rate_first": sr_first,
            "n_strict_pairs": n_strict,
            "n_equal_pairs": n_equal,
            "n_equal_pairs_noab": n_equal_noab,
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(out_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"Wrote {len(out_df)} rows -> {out_path}")


if __name__ == "__main__":
    main()


