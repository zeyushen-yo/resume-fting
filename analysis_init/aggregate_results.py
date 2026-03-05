#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
import sys
from dataclasses import dataclass
import ast
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Aggregate evaluation results into accuracy tables")
    p.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="One or more CSV files or directories containing eval CSVs",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="/home/zs7353/resume_validity/analysis/aggregates",
        help="Directory to write aggregated CSVs",
    )
    p.add_argument(
        "--format",
        type=str,
        default="json",
        choices=["json", "csv"],
        help="Output format (default: json)",
    )
    p.add_argument(
        "--include_substrings",
        type=str,
        nargs="*",
        default=[],
        help="Only include input CSVs whose path contains ANY of these substrings (OR semantics)",
    )
    p.add_argument(
        "--exclude_substrings",
        type=str,
        nargs="*",
        default=[],
        help="Exclude input CSVs whose path contains ANY of these substrings",
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


def _filter_files(files: List[Path], includes: List[str], excludes: List[str]) -> List[Path]:
    if not includes and not excludes:
        return files
    out: List[Path] = []
    for fp in files:
        s = str(fp)
        ok_include = True if not includes else any(sub in s for sub in includes)
        ok_exclude = not any(sub in s for sub in excludes)
        if ok_include and ok_exclude:
            out.append(fp)
    return out


def _infer_framework_from_path(path: Path) -> str:
    # Heuristic: parent directory name contains 'agentic' or 'baseline'
    s = str(path).lower()
    if "/agentic/" in s:
        return "agentic"
    if "/baseline/" in s:
        return "baseline"
    return "unknown"


def _infer_model_from_filename(path: Path) -> str:
    # Filename pattern: <model_short>_paired_resume_decisions_...
    stem = path.stem
    if "_paired_resume_decisions_" in stem:
        return stem.split("_paired_resume_decisions_", 1)[0]
    return stem


def _coerce_demographics(row: Dict[str, Any]) -> Tuple[str, str]:
    db = str(row.get("demographic_base") or "").strip()
    dv = str(row.get("demographic_variant") or "").strip()
    if db and dv:
        return db, dv
    dem = row.get("demographics")
    if isinstance(dem, (list, tuple)) and len(dem) >= 2:
        return str(dem[0]), str(dem[1])
    # Fallback: attempt JSON parse if it's a string
    if isinstance(dem, str):
        # Try JSON first
        try:
            arr = json.loads(dem)
            if isinstance(arr, (list, tuple)) and len(arr) >= 2:
                return str(arr[0]), str(arr[1])
        except Exception:
            pass
        # Then try Python literal (handles single-quoted lists)
        try:
            arr = ast.literal_eval(dem)
            if isinstance(arr, (list, tuple)) and len(arr) >= 2:
                return str(arr[0]), str(arr[1])
        except Exception:
            pass
    return "", ""


def _split_bullets(s: str) -> List[str]:
    s = s.replace("\r", "\n")
    # Split on newlines, bullets, semicolons, or vertical bars
    parts = re.split(r"[\n]+|[•\u2022]+|;|\s\|\s", s)
    out: List[str] = []
    for p in parts:
        t = p.strip().lstrip("-•*+>\t ").rstrip()
        if t:
            out.append(t)
    return out


def _coerce_diff_quals(val: Any) -> List[str]:
    # Normalize differed_qualifications to a list of full qualification strings
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        # JSON list
        if s.startswith("[") and s.endswith("]"):
            try:
                j = json.loads(s)
                if isinstance(j, list):
                    return [str(x).strip() for x in j if str(x).strip()]
            except Exception:
                pass
            # Python literal list (e.g., single quotes)
            try:
                j = ast.literal_eval(s)
                if isinstance(j, list):
                    return [str(x).strip() for x in j if str(x).strip()]
            except Exception:
                pass
        # Otherwise, split on common bullet separators
        toks = _split_bullets(s)
        return [t for t in toks if t]
    return []


def _normalize_rows(df: pd.DataFrame, framework: str, model_id: str) -> pd.DataFrame:
    df = df.copy()
    df["framework"] = framework
    df["model_id"] = model_id

    # experiment_type normalization
    if "experiment_type_norm" in df.columns:
        df["experiment_type"] = df["experiment_type_norm"].fillna(df.get("experiment_type", ""))
    else:
        df["experiment_type"] = df.get("experiment_type", "")
    df["experiment_type"] = df["experiment_type"].astype(str).str.lower()

    # demographics
    if "demographic_base" not in df.columns:
        df["demographic_base"] = ""
    if "demographic_variant" not in df.columns:
        df["demographic_variant"] = ""
    # Fill missing from `demographics`
    if (df["demographic_base"] == "").any() or (df["demographic_variant"] == "").any():
        bases: List[str] = []
        vars_: List[str] = []
        for _, r in df.iterrows():
            b, v = _coerce_demographics(dict(r))
            bases.append(b)
            vars_.append(v)
        df["demographic_base"] = df["demographic_base"].mask(df["demographic_base"]=="", bases)
        df["demographic_variant"] = df["demographic_variant"].mask(df["demographic_variant"]=="", vars_)

    # pair_type normalization
    if "pair_type" in df.columns:
        df["pair_type"] = df["pair_type"].astype(str).str.lower()
    else:
        df["pair_type"] = ""

    # num_differed coercion
    if "num_differed" in df.columns:
        df["num_differed"] = pd.to_numeric(df["num_differed"], errors="coerce").fillna(0).astype(int)
    else:
        df["num_differed"] = 0

    # is_valid coercion
    if "is_valid" in df.columns:
        df["is_valid"] = df["is_valid"].astype(str).str.lower().isin(["true", "1", "yes", "y", "t"])
    else:
        df["is_valid"] = False

    # differed_qualifications explode
    if "differed_qualifications" in df.columns:
        df["diff_quals_list"] = df["differed_qualifications"].apply(_coerce_diff_quals)
    else:
        df["diff_quals_list"] = [[] for _ in range(len(df))]

    return df


def _agg_accuracy(g: pd.core.groupby.generic.DataFrameGroupBy) -> pd.DataFrame:
    out = g["is_valid"].agg(n="count", accuracy="mean").reset_index()
    # Ensure float
    out["accuracy"] = out["accuracy"].astype(float).fillna(0.0)
    return out


def _write(df: pd.DataFrame, out_path: Path, fmt: str = "json") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(out_path, index=False, quoting=csv.QUOTE_ALL)
    else:
        # JSON array of records (pretty-printed for readability)
        records = df.to_dict(orient="records")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(df)} rows -> {out_path}")


def aggregate(all_files: List[Path], out_dir: Path, fmt: str = "json") -> None:
    records: List[pd.DataFrame] = []
    for fp in all_files:
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        framework = _infer_framework_from_path(fp)
        model_id = _infer_model_from_filename(fp)
        df = _normalize_rows(df, framework=framework, model_id=model_id)
        records.append(df)

    if not records:
        print("No input rows found.")
        return

    df_all = pd.concat(records, ignore_index=True)

    # Split by experiment types
    df_valid = df_all[df_all["experiment_type"] == "validity"]
    df_valid_demo = df_all[df_all["experiment_type"] == "validity_demographics"]
    df_fair  = df_all[df_all["experiment_type"] == "fairness"]
    df_impl  = df_all[df_all["experiment_type"] == "implicit_demographics_fairness"]

    # validity_1: overall accuracy per model/framework
    g = df_valid.groupby(["model_id", "framework"], dropna=False)
    validity_overall = _agg_accuracy(g)
    _write(validity_overall, out_dir / ("validity_overall." + fmt), fmt)

    # validity_2: by job_title
    if "job_title" not in df_valid.columns:
        df_valid["job_title"] = df_valid.get("job_title", "")
    g = df_valid.groupby(["model_id", "framework", "job_title"], dropna=False)
    validity_by_job = _agg_accuracy(g)
    _write(validity_by_job, out_dir / ("validity_by_job_title." + fmt), fmt)

    # validity_3: by pair_type and num_differed
    g = df_valid.groupby(["model_id", "framework", "pair_type", "num_differed"], dropna=False)
    validity_by_pair_num = _agg_accuracy(g)
    _write(validity_by_pair_num, out_dir / ("validity_by_pair_type_num_differed." + fmt), fmt)

    # validity_4: by each differed_qualification (explode)
    df_v_ex = df_valid.explode("diff_quals_list")
    df_v_ex = df_v_ex[df_v_ex["diff_quals_list"].astype(str) != ""]
    g = df_v_ex.groupby(["model_id", "framework", "diff_quals_list"], dropna=False)
    validity_by_diffqual = _agg_accuracy(g).rename(columns={"diff_quals_list": "differed_qualification"})
    _write(validity_by_diffqual, out_dir / ("validity_by_differed_qualification." + fmt), fmt)

    # validity_demographics: mirror of validity aggregations
    if not df_valid_demo.empty:
        g = df_valid_demo.groupby(["model_id", "framework"], dropna=False)
        validity_demo_overall = _agg_accuracy(g)
        _write(validity_demo_overall, out_dir / ("validity_demographics_overall." + fmt), fmt)

        if "job_title" not in df_valid_demo.columns:
            df_valid_demo["job_title"] = df_valid_demo.get("job_title", "")
        g = df_valid_demo.groupby(["model_id", "framework", "job_title"], dropna=False)
        validity_demo_by_job = _agg_accuracy(g)
        _write(validity_demo_by_job, out_dir / ("validity_demographics_by_job_title." + fmt), fmt)

        g = df_valid_demo.groupby(["model_id", "framework", "pair_type", "num_differed"], dropna=False)
        validity_demo_by_pair_num = _agg_accuracy(g)
        _write(validity_demo_by_pair_num, out_dir / ("validity_demographics_by_pair_type_num_differed." + fmt), fmt)

        df_vd_ex = df_valid_demo.explode("diff_quals_list")
        df_vd_ex = df_vd_ex[df_vd_ex["diff_quals_list"].astype(str) != ""]
        g = df_vd_ex.groupby(["model_id", "framework", "diff_quals_list"], dropna=False)
        validity_demo_by_diffqual = _agg_accuracy(g).rename(columns={"diff_quals_list": "differed_qualification"})
        _write(validity_demo_by_diffqual, out_dir / ("validity_demographics_by_differed_qualification." + fmt), fmt)

    # fairness_1: accuracy by demographic pair
    g = df_fair.groupby(["model_id", "framework", "demographic_base", "demographic_variant"], dropna=False)
    fair_by_demo = _agg_accuracy(g)
    _write(fair_by_demo, out_dir / ("fairness_by_demographics." + fmt), fmt)

    # fairness_2: by demo x job_title
    g = df_fair.groupby(["model_id", "framework", "demographic_base", "demographic_variant", "job_title"], dropna=False)
    fair_by_demo_job = _agg_accuracy(g)
    _write(fair_by_demo_job, out_dir / ("fairness_by_demographics_job_title." + fmt), fmt)

    # fairness_3: by demo x pair_type x num_differed
    g = df_fair.groupby(["model_id", "framework", "demographic_base", "demographic_variant", "pair_type", "num_differed"], dropna=False)
    fair_by_demo_pair_num = _agg_accuracy(g)
    _write(fair_by_demo_pair_num, out_dir / ("fairness_by_demographics_pair_type_num_differed." + fmt), fmt)

    # implicit fairness 1/2/3
    g = df_impl.groupby(["model_id", "framework", "demographic_base", "demographic_variant"], dropna=False)
    impl_by_demo = _agg_accuracy(g)
    _write(impl_by_demo, out_dir / ("implicit_fairness_by_demographics." + fmt), fmt)

    g = df_impl.groupby(["model_id", "framework", "demographic_base", "demographic_variant", "job_title"], dropna=False)
    impl_by_demo_job = _agg_accuracy(g)
    _write(impl_by_demo_job, out_dir / ("implicit_fairness_by_demographics_job_title." + fmt), fmt)

    g = df_impl.groupby(["model_id", "framework", "demographic_base", "demographic_variant", "pair_type", "num_differed"], dropna=False)
    impl_by_demo_pair_num = _agg_accuracy(g)
    _write(impl_by_demo_pair_num, out_dir / ("implicit_fairness_by_demographics_pair_type_num_differed." + fmt), fmt)


def main() -> None:
    args = parse_args()
    files = _gather_csv_files(args.inputs)
    files = _filter_files(files, args.include_substrings, args.exclude_substrings)
    if not files:
        print("No CSV inputs found.")
        sys.exit(1)
    aggregate(files, Path(args.output_dir), fmt=args.format)


if __name__ == "__main__":
    main()


