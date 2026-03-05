#!/usr/bin/env python3
"""
Compute selection-rate disaggregations (gender, race) for the non-CS job runs.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


TARGET_JOBS = {
    "Financial Analyst",
    "Nurse Practitioner",
    "Wind Turbine Technician",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Aggregate selection-rate by demographic group for non-CS jobs")
    p.add_argument("--noabstain_dir", type=str, required=True, help="Directory with evaluate_model_no_abstain CSVs")
    p.add_argument("--out_gender_csv", type=str, default="analysis/noncs_selection_rate_gender.csv")
    p.add_argument("--out_race_csv", type=str, default="analysis/noncs_selection_rate_race.csv")
    return p.parse_args()


def load_equal_pairs(eval_dir: Path) -> pd.DataFrame:
    recs: List[pd.DataFrame] = []
    for fp in sorted(eval_dir.glob("*.csv")):
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        model_id = fp.name.split("_paired_resume")[0]
        df["model_id"] = model_id
        df["job_type"] = df.get("job_title", "").astype(str).str.strip()
        recs.append(df)
    if not recs:
        return pd.DataFrame()
    all_df = pd.concat(recs, ignore_index=True)
    all_df = all_df[all_df["job_type"].isin(TARGET_JOBS)].copy()
    all_df = all_df[all_df["num_differed"].fillna(0) == 0]
    all_df["decision"] = all_df["decision"].astype(str).str.lower()
    return all_df


def symmetrized_selection(df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    group_cols = ["job_type", "model_id"]
    for (job, model), g in df.groupby(group_cols):
        grp = g.copy()
        grp["pair_key"] = grp.apply(
            lambda r: " / ".join(sorted([str(r["demographic_base"]), str(r["demographic_variant"])])),
            axis=1,
        )
        for pair_key, rows in grp.groupby("pair_key"):
            counts: Dict[str, Dict[str, float]] = defaultdict(lambda: {"sel": 0.0, "total": 0.0})
            for _, row in rows.iterrows():
                base = str(row["demographic_base"])
                variant = str(row["demographic_variant"])
                decision = row["decision"]
                if decision == "first":
                    counts[base]["sel"] += 1
                elif decision == "second":
                    counts[variant]["sel"] += 1
                # Both candidates appeared once in this presentation
                counts[base]["total"] += 1
                counts[variant]["total"] += 1
            for group_code, stats in counts.items():
                total = stats["total"]
                if total <= 0:
                    continue
                sel_rate = stats["sel"] / total
                records.append({
                    "job_type": job,
                    "model_id": model,
                    "pair_key": pair_key,
                    "group": group_code,
                    "selection_rate": sel_rate,
                    "den": total,
                })
    return pd.DataFrame(records)


def map_gender(group_code: str) -> str | None:
    parts = group_code.split("_")
    if len(parts) != 2:
        return None
    return {"M": "Men", "W": "Women"}.get(parts[1])


def map_race(group_code: str) -> str | None:
    parts = group_code.split("_")
    if len(parts) != 2:
        return None
    return {"B": "Black", "W": "White"}.get(parts[0])


def aggregate_dimension(df: pd.DataFrame, mapper, col_name: str) -> pd.DataFrame:
    d = df.copy()
    d[col_name] = d["group"].map(mapper)
    d = d[d[col_name].notna()].copy()
    if d.empty:
        return d
    agg = d.groupby(["job_type", "model_id", col_name], as_index=False).apply(
        lambda g: pd.Series({
            "selection_rate": (g["selection_rate"] * g["den"]).sum() / g["den"].sum(),
            "den": g["den"].sum(),
        })
    ).reset_index(drop=True)
    agg = agg.rename(columns={col_name: "group"})
    return agg


def main() -> None:
    args = parse_args()
    noab_dir = Path(args.noabstain_dir)
    gender_out = Path(args.out_gender_csv)
    race_out = Path(args.out_race_csv)
    gender_out.parent.mkdir(parents=True, exist_ok=True)
    race_out.parent.mkdir(parents=True, exist_ok=True)

    df = load_equal_pairs(noab_dir)
    if df.empty:
        raise RuntimeError(f"No equal-pair rows found under {noab_dir}")
    sym = symmetrized_selection(df)
    if sym.empty:
        raise RuntimeError("No selection-rate records generated; check input data.")
    gender = aggregate_dimension(sym, map_gender, "gender")
    race = aggregate_dimension(sym, map_race, "race")
    gender.to_csv(gender_out, index=False)
    race.to_csv(race_out, index=False)
    print(f"Wrote {len(gender)} rows -> {gender_out}")
    print(f"Wrote {len(race)} rows -> {race_out}")


if __name__ == "__main__":
    main()



