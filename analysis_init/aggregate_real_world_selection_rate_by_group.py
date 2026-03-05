#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Aggregate real-world selection rate by gender/race (symmetrized) from no-abstain evals")
    p.add_argument("--noabstain_dir", type=str, default="/home/zs7353/resume_validity/evaluations/baseline_noabstain")
    p.add_argument("--pairs_root", type=str, default="/home/zs7353/resume_validity/data/pairs_from_real_world")
    p.add_argument("--out_gender_csv", type=str, default="/home/zs7353/resume_validity/analysis/real_world_selection_rate_gender.csv")
    p.add_argument("--out_race_csv", type=str, default="/home/zs7353/resume_validity/analysis/real_world_selection_rate_race.csv")
    return p.parse_args()


def load_allowed_jobs(pairs_root: Path) -> set[Tuple[str, str]]:
    allowed: set[Tuple[str, str]] = set()
    for pf in pairs_root.rglob("pairs_shard_00_of_01.jsonl"):
        try:
            with open(pf, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    js = obj.get("job_source") or {}
                    comp = str(js.get("company") or "").strip()
                    title = str(js.get("title") or "").strip()
                    if comp or title:
                        allowed.add((comp, title))
                    break
        except Exception:
            continue
    return allowed


def load_noab_evals(noab_dir: Path, allowed: set[Tuple[str, str]]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for fp in sorted(noab_dir.glob("*.csv")):
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        if df.empty:
            continue
        # Parse job_source (stringified dict)
        companies: List[str] = []
        titles: List[str] = []
        for _, r in df.iterrows():
            js = r.get("job_source")
            comp, title = "", ""
            if isinstance(js, str) and js.strip().startswith("{"):
                try:
                    d = ast.literal_eval(js)
                    comp = str(d.get("company") or "").strip()
                    title = str(d.get("title") or "").strip()
                except Exception:
                    pass
            companies.append(comp)
            titles.append(title)
        df["job_company"] = companies
        df["job_title_posting"] = titles
        if allowed:
            df = df[df.apply(lambda r: (str(r["job_company"]), str(r["job_title_posting"])) in allowed, axis=1)]
        if df.empty:
            continue
        # Normalize
        df["model_id"] = str(fp.name).split("_paired_resume_decisions_", 1)[0]
        df["decision"] = df.get("decision", "").astype(str).str.lower()
        df["demographic_base"] = df.get("demographic_base", "").astype(str)
        df["demographic_variant"] = df.get("demographic_variant", "").astype(str)
        df["experiment_type"] = df.get("experiment_type_norm", df.get("experiment_type", "")).astype(str).str.lower()
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def symmetrized_rates(df: pd.DataFrame) -> pd.DataFrame:
    # Only equal pairs (fairness / implicit_demographics_fairness)
    e = df[df["experiment_type"].isin(["fairness", "implicit_demographics_fairness"])].copy()
    if e.empty:
        return pd.DataFrame()
    recs: List[Dict[str, Any]] = []
    for (model_id, comp, title), g in e.groupby(["model_id", "job_company", "job_title_posting"]):
        # Aggregate across ALL pairs for this (job, model)
        # For each demographic code, count how many times they were selected vs appeared IN VALID DECISIONS
        counts: Dict[str, Dict[str, float]] = {}
        for _, r in g.iterrows():
            base = r["demographic_base"]
            var = r["demographic_variant"]
            d = r["decision"]
            # Only count pairs with valid decisions (first or second)
            if d not in ("first", "second"):
                continue
            # Each person appears once in this pair
            for code in (base, var):
                counts.setdefault(code, {"sel": 0.0, "tot": 0.0})
                counts[code]["tot"] += 1.0
            # Credit selection
            if d == "first":
                counts[base]["sel"] += 1.0
            elif d == "second":
                counts[var]["sel"] += 1.0
        # Now compute rates
        for code, st in counts.items():
            if st["tot"] <= 0:
                continue
            recs.append({
                "model_id": model_id,
                "job_company": comp,
                "job_title_posting": title,
                "group_code": code,
                "selection_rate": st["sel"] / st["tot"],
                "den": st["tot"],
            })
    return pd.DataFrame(recs)


def map_gender(code: str) -> str | None:
    try:
        _, g = code.split("_", 1)
        return {"M": "Men", "W": "Women"}.get(g)
    except Exception:
        return None


def map_race(code: str) -> str | None:
    try:
        r, _ = code.split("_", 1)
        return {"B": "Black", "W": "White"}.get(r)
    except Exception:
        return None


def aggregate_dimension(df: pd.DataFrame, mapper, col_name: str) -> pd.DataFrame:
    d = df.copy()
    d[col_name] = d["group_code"].map(mapper)
    d = d[d[col_name].notna()].copy()
    if d.empty:
        return d
    agg = d.groupby(["job_company", "job_title_posting", "model_id", col_name], as_index=False).apply(
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
    allowed = load_allowed_jobs(Path(args.pairs_root))
    df = load_noab_evals(noab_dir, allowed)
    if df.empty:
        raise RuntimeError("No evaluation rows found for real-world pairs.")
    sym = symmetrized_rates(df)
    if sym.empty:
        raise RuntimeError("No symmetrized rows computed.")
    gender = aggregate_dimension(sym, map_gender, "gender")
    race = aggregate_dimension(sym, map_race, "race")
    Path(args.out_gender_csv).parent.mkdir(parents=True, exist_ok=True)
    gender.to_csv(args.out_gender_csv, index=False)
    race.to_csv(args.out_race_csv, index=False)
    print(f"Wrote {len(gender)} rows -> {args.out_gender_csv}")
    print(f"Wrote {len(race)} rows -> {args.out_race_csv}")


if __name__ == "__main__":
    main()



