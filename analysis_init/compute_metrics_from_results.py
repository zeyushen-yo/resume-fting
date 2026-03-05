#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Compute evaluation metrics from result CSVs (with/without abstain)")
    p.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        default=[
            "/home/zs7353/resume_validity/evaluations/baseline",
            "/home/zs7353/resume_validity/evaluations/baseline_noabstain",
        ],
        help="One or more CSV files or directories containing eval CSVs",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="/home/zs7353/resume_validity/analysis/metrics_brainstormed",
        help="Directory to write metric JSONs",
    )
    p.add_argument(
        "--format",
        type=str,
        default="json",
        choices=["json"],
        help="Output format (currently only json)",
    )
    return p.parse_args()


def _gather_csv_files(paths: List[str]) -> List[Path]:
    files: List[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.csv")))
        elif path.is_file() and path.suffix.lower() == ".csv":
            files.append(path)
    return files


def _infer_framework_from_path(path: Path) -> str:
    s = str(path).lower()
    if "/baseline_noabstain/" in s:
        return "baseline_noabstain"
    if "/baseline/" in s:
        return "baseline"
    return "unknown"


def _infer_model_from_filename(path: Path) -> str:
    stem = path.stem
    if "_paired_resume_decisions_" in stem:
        return stem.split("_paired_resume_decisions_", 1)[0]
    return stem


def _infer_constructed_by(path: Path) -> str:
    s = path.name.lower()
    # Primary tags in filenames
    if "_claude_" in s:
        return "claude"
    if "_gemini_" in s:
        return "gemini"
    # Default policy consistent with existing analysis: generic rel6/v_k shards are gemini unless explicitly tagged as claude
    if "_rel6_shard" in s or "_v_k_shard" in s:
        return "gemini"
    return "unknown"


def _coerce_demographics(row: Dict[str, Any]) -> Tuple[str, str]:
    db = str(row.get("demographic_base") or "").strip()
    dv = str(row.get("demographic_variant") or "").strip()
    if db and dv:
        return db, dv
    dem = row.get("demographics")
    if isinstance(dem, (list, tuple)) and len(dem) >= 2:
        return str(dem[0]), str(dem[1])
    if isinstance(dem, str):
        try:
            arr = json.loads(dem)
            if isinstance(arr, (list, tuple)) and len(arr) >= 2:
                return str(arr[0]), str(arr[1])
        except Exception:
            pass
        try:
            arr = ast.literal_eval(dem)
            if isinstance(arr, (list, tuple)) and len(arr) >= 2:
                return str(arr[0]), str(arr[1])
        except Exception:
            pass
    return "", ""


def _split_bullets(s: str) -> List[str]:
    s = s.replace("\r", "\n")
    parts = pd.Series(s).str.split(r"[\n]+|[•\u2022]+|;|\s\|\s", regex=True).iat[0] or []
    out: List[str] = []
    for p in parts:
        t = str(p).strip().lstrip("-•*+>\t ").rstrip()
        if t:
            out.append(t)
    return out


def _coerce_diff_quals(val: Any) -> List[str]:
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                j = json.loads(s)
                if isinstance(j, list):
                    return [str(x).strip() for x in j if str(x).strip()]
            except Exception:
                pass
            try:
                j = ast.literal_eval(s)
                if isinstance(j, list):
                    return [str(x).strip() for x in j if str(x).strip()]
            except Exception:
                pass
        toks = _split_bullets(s)
        return [t for t in toks if t]
    return []


def _normalize_df(df: pd.DataFrame, framework: str, model_id: str, constructed_by: str) -> pd.DataFrame:
    d = df.copy()
    d["framework"] = framework
    d["model_id"] = model_id
    d["constructed_by"] = constructed_by

    # experiment type
    if "experiment_type_norm" in d.columns:
        d["experiment_type"] = d["experiment_type_norm"].fillna(d.get("experiment_type", ""))
    else:
        d["experiment_type"] = d.get("experiment_type", "")
    d["experiment_type"] = d["experiment_type"].astype(str).str.lower()

    # demographics
    if "demographic_base" not in d.columns:
        d["demographic_base"] = ""
    if "demographic_variant" not in d.columns:
        d["demographic_variant"] = ""
    if (d["demographic_base"] == "").any() or (d["demographic_variant"] == "").any():
        bases: List[str] = []
        vars_: List[str] = []
        for _, r in d.iterrows():
            b, v = _coerce_demographics(dict(r))
            bases.append(b)
            vars_.append(v)
        d["demographic_base"] = d["demographic_base"].mask(d["demographic_base"] == "", bases)
        d["demographic_variant"] = d["demographic_variant"].mask(d["demographic_variant"] == "", vars_)

    # pair_type
    if "pair_type" in d.columns:
        d["pair_type"] = d["pair_type"].astype(str).str.lower()
    else:
        d["pair_type"] = ""

    # num_differed
    if "num_differed" in d.columns:
        d["num_differed"] = pd.to_numeric(d["num_differed"], errors="coerce").fillna(0).astype(int)
    else:
        d["num_differed"] = 0

    # is_valid
    if "is_valid" in d.columns:
        d["is_valid"] = d["is_valid"].astype(str).str.lower().isin(["true", "1", "yes", "y", "t"])
    else:
        d["is_valid"] = False

    # abstained
    if "abstained" in d.columns:
        d["abstained"] = d["abstained"].astype(str).str.lower().isin(["true", "1", "yes", "y", "t"])
    else:
        # If column missing (e.g., forced-decision), treat as False
        d["abstained"] = False

    # decision
    if "decision" in d.columns:
        d["decision"] = d["decision"].astype(str).str.lower().fillna("")
    else:
        d["decision"] = ""

    # differed_qualifications list
    if "differed_qualifications" in d.columns:
        d["diff_quals_list"] = d["differed_qualifications"].apply(_coerce_diff_quals)
    else:
        d["diff_quals_list"] = [[] for _ in range(len(d))]

    return d


def _write_json(records: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(records)} rows -> {out_path}")


def compute_metrics(df_all: pd.DataFrame, out_dir: Path) -> None:
    # Split by experiment and define S (strict preference) and E (equal) sets
    df_valid = df_all[df_all["experiment_type"].isin(["validity", "validity_demographics"])].copy()
    df_valid["better"] = df_valid.get("better", "").astype(str).str.lower()
    # Strict preference: better is first or second and k>=1
    S = df_valid[df_valid["better"].isin(["first", "second"]) & (df_valid["num_differed"] >= 1)].copy()

    df_equal = df_all[df_all["experiment_type"].isin(["fairness", "implicit_demographics_fairness"])].copy()
    # Equally-qualified ties: k==0 (should hold for fairness/implicit fairness)
    E = df_equal[df_equal["num_differed"] == 0].copy()

    group_keys = ["model_id", "framework", "constructed_by"]

    # ---------------------------
    # Criterion Validity (on S)
    # ---------------------------
    crit_overall: List[Dict[str, Any]] = []
    for (model_id, framework, src), g in S.groupby(group_keys, dropna=False):
        n = int(len(g))
        dec = g["decision"].astype(str).str.lower()
        gt = g["better"].astype(str).str.lower()
        correct = ((dec == "first") & (gt == "first")) | ((dec == "second") & (gt == "second"))
        crit_overall.append({
            "model_id": model_id,
            "framework": framework,
            "constructed_by": src,
            "n": n,
            "criterion_validity": float(correct.mean()) if n > 0 else float("nan"),
        })
    _write_json(crit_overall, out_dir / "criterion_validity_overall.json")

    crit_by_k: List[Dict[str, Any]] = []
    for (model_id, framework, src, k), g in S.groupby(group_keys + ["num_differed"], dropna=False):
        n = int(len(g))
        dec = g["decision"].astype(str).str.lower()
        gt = g["better"].astype(str).str.lower()
        correct = ((dec == "first") & (gt == "first")) | ((dec == "second") & (gt == "second"))
        crit_by_k.append({
            "model_id": model_id,
            "framework": framework,
            "constructed_by": src,
            "k": int(k),
            "n": n,
            "criterion_validity": float(correct.mean()) if n > 0 else float("nan"),
        })
    _write_json(crit_by_k, out_dir / "criterion_validity_by_k.json")

    crit_by_demo: List[Dict[str, Any]] = []
    for (model_id, framework, src, db, dv), g in S.groupby(group_keys + ["demographic_base", "demographic_variant"], dropna=False):
        n = int(len(g))
        dec = g["decision"].astype(str).str.lower()
        gt = g["better"].astype(str).str.lower()
        correct = ((dec == "first") & (gt == "first")) | ((dec == "second") & (gt == "second"))
        crit_by_demo.append({
            "model_id": model_id,
            "framework": framework,
            "constructed_by": src,
            "demographic_base": str(db),
            "demographic_variant": str(dv),
            "n": n,
            "criterion_validity": float(correct.mean()) if n > 0 else float("nan"),
        })
    _write_json(crit_by_demo, out_dir / "criterion_validity_by_demographics.json")

    crit_by_demo_k: List[Dict[str, Any]] = []
    for (model_id, framework, src, db, dv, k), g in S.groupby(group_keys + ["demographic_base", "demographic_variant", "num_differed"], dropna=False):
        n = int(len(g))
        dec = g["decision"].astype(str).str.lower()
        gt = g["better"].astype(str).str.lower()
        correct = ((dec == "first") & (gt == "first")) | ((dec == "second") & (gt == "second"))
        crit_by_demo_k.append({
            "model_id": model_id,
            "framework": framework,
            "constructed_by": src,
            "demographic_base": str(db),
            "demographic_variant": str(dv),
            "k": int(k),
            "n": n,
            "criterion_validity": float(correct.mean()) if n > 0 else float("nan"),
        })
    _write_json(crit_by_demo_k, out_dir / "criterion_validity_by_demographics_k.json")

    # ----------------------------------
    # Unjustified Abstentions (on S)
    # ----------------------------------
    uja_overall: List[Dict[str, Any]] = []
    for (model_id, framework, src), g in S.groupby(group_keys, dropna=False):
        n = int(len(g))
        uja_overall.append({
            "model_id": model_id,
            "framework": framework,
            "constructed_by": src,
            "n": n,
            "unjustified_abstention": float(g["abstained"].mean()) if n > 0 else float("nan"),
        })
    _write_json(uja_overall, out_dir / "unjustified_abstention_overall.json")

    uja_by_k: List[Dict[str, Any]] = []
    for (model_id, framework, src, k), g in S.groupby(group_keys + ["num_differed"], dropna=False):
        n = int(len(g))
        uja_by_k.append({
            "model_id": model_id,
            "framework": framework,
            "constructed_by": src,
            "k": int(k),
            "n": n,
            "unjustified_abstention": float(g["abstained"].mean()) if n > 0 else float("nan"),
        })
    _write_json(uja_by_k, out_dir / "unjustified_abstention_by_k.json")

    uja_by_demo: List[Dict[str, Any]] = []
    for (model_id, framework, src, db, dv), g in S.groupby(group_keys + ["demographic_base", "demographic_variant"], dropna=False):
        n = int(len(g))
        uja_by_demo.append({
            "model_id": model_id,
            "framework": framework,
            "constructed_by": src,
            "demographic_base": str(db),
            "demographic_variant": str(dv),
            "n": n,
            "unjustified_abstention": float(g["abstained"].mean()) if n > 0 else float("nan"),
        })
    _write_json(uja_by_demo, out_dir / "unjustified_abstention_by_demographics.json")

    uja_by_demo_k: List[Dict[str, Any]] = []
    for (model_id, framework, src, db, dv, k), g in S.groupby(group_keys + ["demographic_base", "demographic_variant", "num_differed"], dropna=False):
        n = int(len(g))
        uja_by_demo_k.append({
            "model_id": model_id,
            "framework": framework,
            "constructed_by": src,
            "demographic_base": str(db),
            "demographic_variant": str(dv),
            "k": int(k),
            "n": n,
            "unjustified_abstention": float(g["abstained"].mean()) if n > 0 else float("nan"),
        })
    _write_json(uja_by_demo_k, out_dir / "unjustified_abstention_by_demographics_k.json")

    # ----------------------------------
    # Discriminant Validity (on E)
    # ----------------------------------
    dv_overall: List[Dict[str, Any]] = []
    for (model_id, framework, src), g in E.groupby(group_keys, dropna=False):
        n = int(len(g))
        dv_overall.append({
            "model_id": model_id,
            "framework": framework,
            "constructed_by": src,
            "n": n,
            "discriminant_validity": float(g["abstained"].mean()) if n > 0 else float("nan"),
        })
    _write_json(dv_overall, out_dir / "discriminant_validity_overall.json")

    dv_by_k: List[Dict[str, Any]] = []
    for (model_id, framework, src, k), g in E.groupby(group_keys + ["num_differed"], dropna=False):
        n = int(len(g))
        dv_by_k.append({
            "model_id": model_id,
            "framework": framework,
            "constructed_by": src,
            "k": int(k),
            "n": n,
            "discriminant_validity": float(g["abstained"].mean()) if n > 0 else float("nan"),
        })
    _write_json(dv_by_k, out_dir / "discriminant_validity_by_k.json")

    dv_by_demo: List[Dict[str, Any]] = []
    for (model_id, framework, src, db, dv), g in E.groupby(group_keys + ["demographic_base", "demographic_variant"], dropna=False):
        n = int(len(g))
        dv_by_demo.append({
            "model_id": model_id,
            "framework": framework,
            "constructed_by": src,
            "demographic_base": str(db),
            "demographic_variant": str(dv),
            "n": n,
            "discriminant_validity": float(g["abstained"].mean()) if n > 0 else float("nan"),
        })
    _write_json(dv_by_demo, out_dir / "discriminant_validity_by_demographics.json")

    dv_by_demo_k: List[Dict[str, Any]] = []
    for (model_id, framework, src, db, dv, k), g in E.groupby(group_keys + ["demographic_base", "demographic_variant", "num_differed"], dropna=False):
        n = int(len(g))
        dv_by_demo_k.append({
            "model_id": model_id,
            "framework": framework,
            "constructed_by": src,
            "demographic_base": str(db),
            "demographic_variant": str(dv),
            "k": int(k),
            "n": n,
            "discriminant_validity": float(g["abstained"].mean()) if n > 0 else float("nan"),
        })
    _write_json(dv_by_demo_k, out_dir / "discriminant_validity_by_demographics_k.json")

    # ----------------------------------
    # Selection Rate (on E) — tie-breaking bias
    # ----------------------------------
    # We report both over all E and conditional on non-abstentions (useful for baseline with abstain)
    sr_overall: List[Dict[str, Any]] = []
    for (model_id, framework, src), g in E.groupby(group_keys, dropna=False):
        n_all = int(len(g))
        n_nonab = int((~g["abstained"]).sum())
        dec = g["decision"].astype(str).str.lower()
        sr_all = float((dec == "first").mean()) if n_all > 0 else float("nan")
        sr_nonab = float(((dec == "first") & (~g["abstained"])).sum() / n_nonab) if n_nonab > 0 else float("nan")
        sr_overall.append({
            "model_id": model_id,
            "framework": framework,
            "constructed_by": src,
            "n_all": n_all,
            "n_nonab": n_nonab,
            "selection_rate_A_all": sr_all,
            "selection_rate_A_nonab": sr_nonab,
        })
    _write_json(sr_overall, out_dir / "selection_rate_overall.json")

    sr_by_k: List[Dict[str, Any]] = []
    for (model_id, framework, src, k), g in E.groupby(group_keys + ["num_differed"], dropna=False):
        n_all = int(len(g))
        n_nonab = int((~g["abstained"]).sum())
        dec = g["decision"].astype(str).str.lower()
        sr_all = float((dec == "first").mean()) if n_all > 0 else float("nan")
        sr_nonab = float(((dec == "first") & (~g["abstained"])).sum() / n_nonab) if n_nonab > 0 else float("nan")
        sr_by_k.append({
            "model_id": model_id,
            "framework": framework,
            "constructed_by": src,
            "k": int(k),
            "n_all": n_all,
            "n_nonab": n_nonab,
            "selection_rate_A_all": sr_all,
            "selection_rate_A_nonab": sr_nonab,
        })
    _write_json(sr_by_k, out_dir / "selection_rate_by_k.json")

    sr_by_demo: List[Dict[str, Any]] = []
    for (model_id, framework, src, db, dv), g in E.groupby(group_keys + ["demographic_base", "demographic_variant"], dropna=False):
        n_all = int(len(g))
        n_nonab = int((~g["abstained"]).sum())
        dec = g["decision"].astype(str).str.lower()
        sr_all = float((dec == "first").mean()) if n_all > 0 else float("nan")
        sr_nonab = float(((dec == "first") & (~g["abstained"])).sum() / n_nonab) if n_nonab > 0 else float("nan")
        sr_by_demo.append({
            "model_id": model_id,
            "framework": framework,
            "constructed_by": src,
            "demographic_base": str(db),
            "demographic_variant": str(dv),
            "n_all": n_all,
            "n_nonab": n_nonab,
            "selection_rate_A_all": sr_all,
            "selection_rate_A_nonab": sr_nonab,
        })
    _write_json(sr_by_demo, out_dir / "selection_rate_by_demographics.json")

    sr_by_demo_k: List[Dict[str, Any]] = []
    for (model_id, framework, src, db, dv, k), g in E.groupby(group_keys + ["demographic_base", "demographic_variant", "num_differed"], dropna=False):
        n_all = int(len(g))
        n_nonab = int((~g["abstained"]).sum())
        dec = g["decision"].astype(str).str.lower()
        sr_all = float((dec == "first").mean()) if n_all > 0 else float("nan")
        sr_nonab = float(((dec == "first") & (~g["abstained"])).sum() / n_nonab) if n_nonab > 0 else float("nan")
        sr_by_demo_k.append({
            "model_id": model_id,
            "framework": framework,
            "constructed_by": src,
            "demographic_base": str(db),
            "demographic_variant": str(dv),
            "k": int(k),
            "n_all": n_all,
            "n_nonab": n_nonab,
            "selection_rate_A_all": sr_all,
            "selection_rate_A_nonab": sr_nonab,
        })
    _write_json(sr_by_demo_k, out_dir / "selection_rate_by_demographics_k.json")

    # ----------------------------------
    # New: Gender and Race selection rate by Job (symmetrized, order-invariant)
    # ----------------------------------
    def parse_race_gender(code: str) -> Tuple[str, str]:
        try:
            parts = str(code).split("_")
            race = str(parts[0]).strip().upper() if len(parts) >= 1 else ""
            gender = str(parts[1]).strip().upper() if len(parts) >= 2 else ""
            return race, gender
        except Exception:
            return "", ""

    # Work from equal pairs (ties)
    if not E.empty:
        e = E.copy()
        e["decision"] = e["decision"].astype(str).str.lower()
        e["job_title"] = e.get("job_title", "").astype(str)
        # Non-abstain mask for weighting
        e["is_nonab"] = e["decision"].isin(["first", "second"])
        # Parse demographics to race/gender
        e["base_race"], e["base_gender"] = zip(*e["demographic_base"].map(parse_race_gender))
        e["var_race"], e["var_gender"] = zip(*e["demographic_variant"].map(parse_race_gender))

        # Gender selection rate by job (compare M vs W regardless of race)
        gender_rows: List[Dict[str, Any]] = []
        for (model_id, src, job), g in e.groupby(["model_id", "constructed_by", "job_title"], dropna=False):
            # Orientation 1: base gender M vs variant gender W
            g1 = g[(g["base_gender"] == "M") & (g["var_gender"] == "W")].copy()
            w1 = int((g1["is_nonab"]).sum())
            p1 = float(((g1["is_nonab"]) & (g1["decision"] == "first")).sum() / w1) if w1 > 0 else float("nan")
            # Orientation 2: base gender W vs variant gender M
            g2 = g[(g["base_gender"] == "W") & (g["var_gender"] == "M")].copy()
            w2 = int((g2["is_nonab"]).sum())
            p2 = float(((g2["is_nonab"]) & (g2["decision"] == "second")).sum() / w2) if w2 > 0 else float("nan")
            den = w1 + w2
            if den <= 0:
                continue
            # Symmetrized P(male) and P(female)
            p_male = ((0.0 if pd.isna(p1) else p1) * w1 + (0.0 if pd.isna(p2) else p2) * w2) / den
            p_female = 1.0 - p_male
            gender_rows.append({"model_id": model_id, "constructed_by": src, "job_title": job, "group": "M", "selection_rate": p_male, "den": den})
            gender_rows.append({"model_id": model_id, "constructed_by": src, "job_title": job, "group": "W", "selection_rate": p_female, "den": den})
        _write_json(gender_rows, out_dir / "selection_rate_gender_by_job.json")

        # Race selection rate by job (compare B vs W regardless of gender)
        race_rows: List[Dict[str, Any]] = []
        for (model_id, src, job), g in e.groupby(["model_id", "constructed_by", "job_title"], dropna=False):
            # Orientation 1: base race B vs variant race W
            g1 = g[(g["base_race"] == "B") & (g["var_race"] == "W")].copy()
            w1 = int((g1["is_nonab"]).sum())
            p1 = float(((g1["is_nonab"]) & (g1["decision"] == "first")).sum() / w1) if w1 > 0 else float("nan")
            # Orientation 2: base race W vs variant race B
            g2 = g[(g["base_race"] == "W") & (g["var_race"] == "B")].copy()
            w2 = int((g2["is_nonab"]).sum())
            p2 = float(((g2["is_nonab"]) & (g2["decision"] == "second")).sum() / w2) if w2 > 0 else float("nan")
            den = w1 + w2
            if den <= 0:
                continue
            # Symmetrized P(Black) and P(White)
            p_black = ((0.0 if pd.isna(p1) else p1) * w1 + (0.0 if pd.isna(p2) else p2) * w2) / den
            p_white = 1.0 - p_black
            race_rows.append({"model_id": model_id, "constructed_by": src, "job_title": job, "group": "B", "selection_rate": p_black, "den": den})
            race_rows.append({"model_id": model_id, "constructed_by": src, "job_title": job, "group": "W", "selection_rate": p_white, "den": den})
        _write_json(race_rows, out_dir / "selection_rate_race_by_job.json")

        # Criterion validity by job and gender/race (on S): group-conditioned on winner (better) group
        if not S.empty:
            s = S.copy()
            s["job_title"] = s.get("job_title", "").astype(str)
            s["better"] = s["better"].astype(str).str.lower()
            s["is_valid_bool"] = s["is_valid"].astype(bool)
            s["base_race"], s["base_gender"] = zip(*s["demographic_base"].map(parse_race_gender))
            s["var_race"], s["var_gender"] = zip(*s["demographic_variant"].map(parse_race_gender))

            cv_gender_rows: List[Dict[str, Any]] = []
            uja_gender_rows: List[Dict[str, Any]] = []
            for (model_id, src, job), g2 in s.groupby(["model_id", "constructed_by", "job_title"], dropna=False):
                # Cross-gender only
                cg = g2[((g2["base_gender"] == "M") & (g2["var_gender"] == "W")) | ((g2["base_gender"] == "W") & (g2["var_gender"] == "M"))].copy()
                if cg.empty:
                    continue
                # Winner group label
                def winner_gender(row: pd.Series) -> str:
                    if row["better"] == "first":
                        return str(row["base_gender"])
                    if row["better"] == "second":
                        return str(row["var_gender"])
                    return ""
                cg["winner_gender"] = cg.apply(winner_gender, axis=1)
                for grp in ["M", "W"]:
                    sub = cg[cg["winner_gender"] == grp]
                    den = int(len(sub))
                    if den <= 0:
                        continue
                    cv = float(sub["is_valid_bool"].mean()) if den > 0 else float("nan")
                    uja = float(sub["abstained"].mean()) if den > 0 else float("nan")
                    cv_gender_rows.append({"model_id": model_id, "constructed_by": src, "job_title": job, "group": grp, "rate": cv, "den": den})
                    uja_gender_rows.append({"model_id": model_id, "constructed_by": src, "job_title": job, "group": grp, "rate": uja, "den": den})
            _write_json(cv_gender_rows, out_dir / "criterion_validity_gender_by_job.json")
            _write_json(uja_gender_rows, out_dir / "unjustified_abstention_gender_by_job.json")

            cv_race_rows: List[Dict[str, Any]] = []
            uja_race_rows: List[Dict[str, Any]] = []
            for (model_id, src, job), g2 in s.groupby(["model_id", "constructed_by", "job_title"], dropna=False):
                # Cross-race only (B vs W)
                cr = g2[((g2["base_race"] == "B") & (g2["var_race"] == "W")) | ((g2["base_race"] == "W") & (g2["var_race"] == "B"))].copy()
                if cr.empty:
                    continue
                # Winner group label
                def winner_race(row: pd.Series) -> str:
                    if row["better"] == "first":
                        return str(row["base_race"])
                    if row["better"] == "second":
                        return str(row["var_race"])
                    return ""
                cr["winner_race"] = cr.apply(winner_race, axis=1)
                for grp in ["B", "W"]:
                    sub = cr[cr["winner_race"] == grp]
                    den = int(len(sub))
                    if den <= 0:
                        continue
                    cv = float(sub["is_valid_bool"].mean()) if den > 0 else float("nan")
                    uja = float(sub["abstained"].mean()) if den > 0 else float("nan")
                    cv_race_rows.append({"model_id": model_id, "constructed_by": src, "job_title": job, "group": grp, "rate": cv, "den": den})
                    uja_race_rows.append({"model_id": model_id, "constructed_by": src, "job_title": job, "group": grp, "rate": uja, "den": den})
            _write_json(cv_race_rows, out_dir / "criterion_validity_race_by_job.json")
            _write_json(uja_race_rows, out_dir / "unjustified_abstention_race_by_job.json")

            # Criterion validity by job with three race-pair categories: W_W, W_B (either orientation), B_B
            cv_race3_rows: List[Dict[str, Any]] = []
            for (model_id, src, job), g2 in s.groupby(["model_id", "constructed_by", "job_title"], dropna=False):
                ww = g2[(g2["base_race"] == "W") & (g2["var_race"] == "W")]
                bb = g2[(g2["base_race"] == "B") & (g2["var_race"] == "B")]
                wb = g2[((g2["base_race"] == "W") & (g2["var_race"] == "B")) | ((g2["base_race"] == "B") & (g2["var_race"] == "W"))]
                for label, sub in [("W_W", ww), ("W_B", wb), ("B_B", bb)]:
                    den = int(len(sub))
                    if den <= 0:
                        continue
                    rate = float(sub["is_valid_bool"].mean())
                    cv_race3_rows.append({
                        "model_id": model_id,
                        "constructed_by": src,
                        "job_title": job,
                        "group": label,
                        "rate": rate,
                        "den": den,
                    })
            _write_json(cv_race3_rows, out_dir / "criterion_validity_race3_by_job.json")

            # Criterion validity / Unjustified abstention by job with three gender-pair categories: M_M, M_W (either orientation), W_W
            cv_gender3_rows: List[Dict[str, Any]] = []
            uja_gender3_rows: List[Dict[str, Any]] = []
            for (model_id, src, job), g2 in s.groupby(["mixer" if False else "model_id", "constructed_by", "job_title"], dropna=False):
                mm = g2[(g2["base_gender"] == "M") & (g2["var_gender"] == "M")]
                ww = g2[(g2["base_gender"] == "W") & (g2["var_gender"] == "W")]
                mw = g2[((g2["base_gender"] == "M") & (g2["var_gender"] == "W")) | ((g2["base_gender"] == "W") & (g2["var_gender"] == "M"))]
                for label, sub in [("M_M", mm), ("M_W", mw), ("W_W", ww)]:
                    den = int(len(sub))
                    if den <= 0:
                        continue
                    cv_val = float(sub["is_valid_bool"].mean())
                    uja_val = float(sub["abstained"].mean())
                    cv_gender3_rows.append({
                        "model_id": model_id,
                        "constructed_by": src,
                        "job_title": job,
                        "group": label,
                        "rate": cv_val,
                        "den": den,
                    })
                    uja_val_dict = {
                        "model_id": model_id,
                        "constructed_by": src,
                        "job_title": job,
                        "group": label,
                        "rate": uja_val,
                        "den": den,
                    }
                    uja_gender3_rows.append(uja_val_dict)
            _write_json(cv_gender3_rows, out_dir / "criterion_validity_gender3_by_job.json")
            _write_json(uja_gender3_rows, out_dir / "unjustified_abstention_gender3_by_job.json")

        # Discriminant validity (on E ties) by job and gender/race (pair-level)
        dv_gender_rows: List[Dict[str, Any]] = []
        dv_race_rows: List[Dict[str, Any]] = []
        for (model_id, src, job), g2 in e.groupby(["model_id", "constructed_by", "job_title"], dropna=False):
            # Gender pair: M vs W
            cg = g2[((g2["base_gender"] == "M") & (g2["var_gender"] == "W")) | ((g2["base_gender"] == "W") & (g2["var_gender"] == "M"))].copy()
            den_g = int(len(cg))
            if den_g > 0:
                dv = float(cg["abstained"].mean())
                dv_gender_rows.append({"model_id": model_id, "constructed_by": src, "job_title": job, "pair": "M_vs_W", "rate": dv, "den": den_g})
            # Race pair: B vs W
            cr = g2[((g2["base_race"] == "B") & (g2["var_race"] == "W")) | ((g2["base_race"] == "W") & (g2["var_race"] == "B"))].copy()
            den_r = int(len(cr))
            if den_r > 0:
                dv = float(cr["abstained"].mean())
                dv_race_rows.append({"model_id": model_id, "constructed_by": src, "job_title": job, "pair": "B_vs_W", "rate": dv, "den": den_r})
        _write_json(dv_gender_rows, out_dir / "discriminant_validity_gender_by_job.json")
        _write_json(dv_race_rows, out_dir / "discriminant_validity_race_by_job.json")

        # Discriminant validity by job with three race-pair categories: W_W, W_B (either orientation), B_B
        dv_race3_rows: List[Dict[str, Any]] = []
        for (model_id, src, job), g2 in e.groupby(["model_id", "constructed_by", "job_title"], dropna=False):
            ww = g2[(g2["base_race"] == "W") & (g2["var_race"] == "W")]
            bb = g2[(g2["base_race"] == "B") & (g2["var_race"] == "B")]
            wb = g2[((g2["base_race"] == "W") & (g2["var_race"] == "B")) | ((g2["base_race"] == "B") & (g2["var_race"] == "W"))]
            for label, sub in [("W_W", ww), ("W_B", wb), ("B_B", bb)]:
                den = int(len(sub))
                if den <= 0:
                    continue
                rate = float(sub["abstained"].mean())
                dv_race3_rows.append({
                    "model_id": model_id,
                    "constructed_by": src,
                    "job_title": job,
                    "group": label,
                    "rate": rate,
                    "den": den,
                })
        _write_json(dv_race3_rows, out_dir / "discriminant_validity_race3_by_job.json")

        # Discriminant validity by job with three gender-pair categories: M_M, M_W (either orientation), W_W
        dv_gender3_rows: List[Dict[str, Any]] = []
        for (model_id, src, job), g2 in e.groupby(["model_id", "constructed_by", "job_title"], dropna=False):
            mm = g2[(g2["base_gender"] == "M") & (g2["var_gender"] == "M")]
            ww = g2[(g2["base_gender"] == "W") & (g2["var_gender"] == "W")]
            mw = g2[((g2["base_gender"] == "M") & (g2["var_gender"] == "W")) | ((g2["base_gender"] == "W") & (g2["var_gender"] == "M"))]
            for label, sub in [("M_M", mm), ("M_W", mw), ("W_W", ww)]:
                den = int(len(sub))
                if den <= 0:
                    continue
                rate = float(sub["abstained"].mean())
                dv_gender3_rows.append({
                    "model_id": model_id,
                    "constructed_by": src,
                    "job_title": job,
                    "group": label,
                    "rate": rate,
                    "den": den,
                })
        _write_json(dv_gender3_rows, out_dir / "discriminant_validity_gender3_by_job.json")

        # Discriminant validity by job and gender/race using same-group equal pairs (two bars: M vs W; B vs W)
        dv_gender_same_rows: List[Dict[str, Any]] = []
        dv_race_same_rows: List[Dict[str, Any]] = []
        for (model_id, src, job), g2 in e.groupby(["model_id", "constructed_by", "job_title"], dropna=False):
            # Gender same-group: M vs M and W vs W
            gm = g2[(g2["base_gender"] == "M") & (g2["var_gender"] == "M")]
            gw = g2[(g2["base_gender"] == "W") & (g2["var_gender"] == "W")]
            den_m = int(len(gm))
            den_w = int(len(gw))
            if den_m > 0:
                dv_m = float(gm["abstained"].mean())
                dv_gender_same_rows.append({"model_id": model_id, "constructed_by": src, "job_title": job, "group": "M", "rate": dv_m, "den": den_m})
            if den_w > 0:
                dv_w = float(gw["abstained"].mean())
                dv_gender_same_rows.append({"model_id": model_id, "constructed_by": src, "job_title": job, "group": "W", "rate": dv_w, "den": den_w})
            # Race same-group: B vs B and W vs W
            rb = g2[(g2["base_race"] == "B") & (g2["var_race"] == "B")]
            rw = g2[(g2["base_race"] == "W") & (g2["var_race"] == "W")]
            den_rb = int(len(rb))
            den_rw = int(len(rw))
            if den_rb > 0:
                dv_b = float(rb["abstained"].mean())
                dv_race_same_rows.append({"model_id": model_id, "constructed_by": src, "job_title": job, "group": "B", "rate": dv_b, "den": den_rb})
            if den_rw > 0:
                dv_w = float(rw["abstained"].mean())
                dv_race_same_rows.append({"model_id": model_id, "constructed_by": src, "job_title": job, "group": "W", "rate": dv_w, "den": den_rw})
        _write_json(dv_gender_same_rows, out_dir / "discriminant_validity_gender_same_by_job.json")
        _write_json(dv_race_same_rows, out_dir / "discriminant_validity_race_same_by_job.json")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    files = _gather_csv_files(args.inputs)
    if not files:
        print("No CSV inputs found.")
        return

    frames: List[pd.DataFrame] = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"[ERROR] Failed to read {fp}: {e}")
            traceback.print_exc()
            continue
        if df.empty:
            continue
        framework = _infer_framework_from_path(fp)
        model_id = _infer_model_from_filename(fp)
        constructed_by = _infer_constructed_by(fp)
        try:
            frames.append(_normalize_df(df, framework=framework, model_id=model_id, constructed_by=constructed_by))
        except Exception as e:
            print(f"[ERROR] Normalization failed for {fp}: {e}")
            traceback.print_exc()
            continue

    if not frames:
        print("No rows after normalization.")
        return

    df_all = pd.concat(frames, ignore_index=True)

    try:
        compute_metrics(df_all, out_dir)
    except Exception as e:
        print(f"[ERROR] Metric computation failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()


