#!/usr/bin/env python3
"""
Aggregate core metrics (criterion validity, unjustified abstention, discriminant
validity, selection rate) for the non-CS job runs we just executed.

Usage:
    python -u analysis/aggregate_noncs_job_metrics.py \
        --abstain_dir evaluations/baseline_noncs \
        --noabstain_dir evaluations/baseline_noabstain \
        --out_csv analysis/noncs_job_metrics.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd


TARGET_JOBS = {
    "Financial Analyst",
    "Nurse Practitioner",
    "Wind Turbine Technician",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Aggregate metrics for non-CS job evaluations")
    p.add_argument("--abstain_dir", type=str, required=True, help="Directory with evaluate_model (abstain) CSVs")
    p.add_argument("--noabstain_dir", type=str, required=True, help="Directory with evaluate_model_no_abstain CSVs")
    p.add_argument("--out_csv", type=str, default="analysis/noncs_job_metrics.csv")
    return p.parse_args()


def load_eval_dir(eval_dir: Path) -> pd.DataFrame:
    recs: List[pd.DataFrame] = []
    for fp in sorted(eval_dir.glob("*.csv")):
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        model_id = fp.name.split("_paired_resume")[0]
        df["model_id"] = model_id
        df["source_file"] = fp.name
        df["job_type"] = df.get("job_title", "").astype(str).str.strip()
        recs.append(df)
    if not recs:
        return pd.DataFrame()
    all_df = pd.concat(recs, ignore_index=True)
    all_df = all_df[all_df["job_type"].isin(TARGET_JOBS)].copy()
    return all_df


def compute_metrics(abstain_df: pd.DataFrame, noab_df: pd.DataFrame) -> pd.DataFrame:
    models = sorted(abstain_df["model_id"].unique().tolist())
    records: List[dict] = []
    for job in sorted(TARGET_JOBS):
        job_abstain = abstain_df[abstain_df["job_type"] == job]
        job_noab = noab_df[noab_df["job_type"] == job]
        for model in models:
            rows = job_abstain[job_abstain["model_id"] == model]
            if rows.empty:
                continue
            strict = rows[rows["num_differed"].fillna(0) > 0]
            equal = rows[rows["num_differed"].fillna(0) == 0]

            n_strict = len(strict)
            n_equal = len(equal)

            crit = float(strict["is_valid"].sum()) / n_strict if n_strict else float("nan")
            unjust = float(strict["abstained"].sum()) / n_strict if n_strict else float("nan")
            discr = float(equal["abstained"].sum()) / n_equal if n_equal else float("nan")

            # Selection rate from no-abstain runs (forced choice on equal pairs)
            rows_noab = job_noab[job_noab["model_id"] == model]
            equal_noab = rows_noab[rows_noab["num_differed"].fillna(0) == 0]
            n_equal_noab = len(equal_noab)
            sel = float((equal_noab["decision"].str.lower() == "first").sum()) / n_equal_noab if n_equal_noab else float("nan")

            records.append({
                "job_type": job,
                "model_id": model,
                "criterion_validity": crit,
                "unjustified_abstention": unjust,
                "discriminant_validity": discr,
                "selection_rate_first": sel,
                "n_strict_pairs": n_strict,
                "n_equal_pairs": n_equal,
                "n_equal_pairs_noab": n_equal_noab,
            })
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    abstain_dir = Path(args.abstain_dir)
    noab_dir = Path(args.noabstain_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    abstain_df = load_eval_dir(abstain_dir)
    noab_df = load_eval_dir(noab_dir)
    if abstain_df.empty:
        raise RuntimeError(f"No evaluation rows found under {abstain_dir}")
    if noab_df.empty:
        raise RuntimeError(f"No evaluation rows found under {noab_dir}")

    summary = compute_metrics(abstain_df, noab_df)
    summary = summary.sort_values(["job_type", "model_id"]).reset_index(drop=True)
    summary.to_csv(out_csv, index=False)
    print(f"Wrote {len(summary)} rows -> {out_csv}")


if __name__ == "__main__":
    main()


