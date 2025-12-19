#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


SHORT_NAMES = {
    "anthropic/claude-sonnet-4": "claude",
    "deepseek/deepseek-chat-v3.1": "deepseek",
    "google/gemini-2.0-flash-001": "gemini-2.0",
    "google/gemini-2.5-pro": "gemini-2.5",
    "google/gemma-3-12b-it": "gemma",
    "openai/gpt-4o-mini": "gpt-4o-mini",
    "openai/gpt-5": "gpt-5",
    "meta-llama/llama-3.1-8b-instruct": "llama-8b",
    "meta-llama/llama-3.3-70b-instruct": "llama-70b",
    # plain ids seen in baseline_noabstain filenames
    "claude-sonnet-4": "claude",
    "deepseek-chat-v3.1": "deepseek",
    "gemini-2.0-flash-001": "gemini-2.0",
    "gemini-2.5-pro": "gemini-2.5",
    "gemma-3-12b-it": "gemma",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-5": "gpt-5",
    "llama-3.1-8b-instruct": "llama-8b",
    "llama-3.3-70b-instruct": "llama-70b",
}


def short_model_name(m: str) -> str:
    return SHORT_NAMES.get(m, m)


def list_csv_files(root: Path, includes: List[str], excludes: List[str]) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*.csv"):
        s = str(p)
        if includes and not any(sub in s for sub in includes):
            continue
        if excludes and any(sub in s for sub in excludes):
            continue
        files.append(p)
    return sorted(files)


def infer_model_from_path(path: Path) -> str:
    stem = path.stem
    if "_paired_resume_decisions_" in stem:
        return stem.split("_paired_resume_decisions_", 1)[0]
    return stem


def eval_file_filters_for_dataset(label: str) -> Tuple[List[str], List[str]]:
    lab = (label or "").strip().lower()
    if lab == "claude":
        # Only pick files produced on Claude dataset
        return ["_claude_noab_"], []
    if lab == "gemini":
        # Only pick files produced on Gemini dataset
        return ["_gemini_noab_"], []
    return ["_noab_"], []


def set_style():
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.55)
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 120,
        "axes.titlesize": 26,
        "axes.titleweight": "bold",
        "axes.labelsize": 22,
        "axes.labelweight": "bold",
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "font.weight": "bold",
    })


def add_ideal_fairness_baseline(ax) -> None:
    y = 0.5
    ax.axhline(y, ls="--", lw=1.2, color="#666666")
    ax.text(0.99, y + 0.01, "ideal fairness", color="#666666", fontsize=10, ha="right", va="bottom", transform=ax.get_yaxis_transform())


def load_noab_rows(root: Path, dataset_label: str) -> pd.DataFrame:
    includes, excludes = eval_file_filters_for_dataset(dataset_label)
    paths = list_csv_files(root, includes=includes, excludes=excludes)
    frames: List[pd.DataFrame] = []
    allowed_groups = {"B_W", "B_M", "W_W", "W_M"}
    for fp in paths:
        try:
            df = pd.read_csv(fp)
        except Exception:
            print(f"[WARN] Failed to read {fp}")
            print(traceback.format_exc())
            continue
        df["model_id"] = infer_model_from_path(fp)
        # Keep fairness and implicit fairness; both are mixed-group pairings
        if "experiment_type_norm" in df.columns:
            et = df["experiment_type_norm"].astype(str)
        else:
            et = df.get("experiment_type", pd.Series([""] * len(df))).astype(str)
        df["experiment_type"] = et.str.lower()
        df = df[df["experiment_type"].isin(["fairness", "implicit_demographics_fairness"])].copy()
        if df.empty:
            continue
        # Required columns
        for c in ["demographic_base", "demographic_variant", "decision"]:
            if c not in df.columns:
                df[c] = ""
        df["demographic_base"] = df["demographic_base"].astype(str)
        df["demographic_variant"] = df["demographic_variant"].astype(str)
        df["decision"] = df["decision"].astype(str).str.lower()
        # Only mixed-group pairs among the four canonical groups
        df = df[(df["demographic_base"] != df["demographic_variant"]) & (df["demographic_base"] != "") & (df["demographic_variant"] != "")]
        df = df[df["demographic_base"].isin(allowed_groups) & df["demographic_variant"].isin(allowed_groups)]
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def compute_selection_rates(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    rows = rows.copy()
    # Normalize decision text and determine selected group
    dec = rows["decision"].astype(str).str.lower().str.strip()
    is_first = dec.str.contains("first")
    is_second = dec.str.contains("second")
    rows["selected_group"] = np.where(is_first, rows["demographic_base"], np.where(is_second, rows["demographic_variant"], ""))
    # Drop any pairs where selection is not one of the canonical groups (Option A strict)
    allowed_groups = {"B_W", "B_M", "W_W", "W_M"}
    rows = rows[rows["selected_group"].isin(allowed_groups)].copy()
    # Denominator: total number of mixed pairs for the model (ensures bars sum to 2 across groups)
    # rate_g = 2 * (# times group selected) / (# mixed pairs for model)
    # SE(rate_g) = 2 * sqrt(p*(1-p)/n_pairs), where p = (# selected)/n_pairs
    all_groups = ["B_W", "B_M", "W_W", "W_M"]
    records = []
    for model_id, mdf in rows.groupby("model_id"):
        n_pairs = len(mdf)
        if n_pairs == 0:
            continue
        for g in all_groups:
            k_sel = int((mdf["selected_group"].astype(str) == g).sum())
            p = k_sel / n_pairs
            rate = 2.0 * p
            se = 2.0 * np.sqrt(p * (1 - p) / n_pairs) if n_pairs > 0 else np.nan
            records.append({"model_id": model_id, "group": str(g), "n": n_pairs, "k": k_sel, "rate": rate, "se": se})
    agg = pd.DataFrame.from_records(records)
    agg["model_short"] = agg["model_id"].map(short_model_name)
    return agg


def plot_selection_rates(agg: pd.DataFrame, out_path: Path, dataset_label: str) -> None:
    if agg.empty:
        print("[WARN] Empty aggregated data; skipping plot")
        return
    set_style()
    # Order models by name and groups by a stable order
    models = sorted(agg["model_id"].unique().tolist())
    agg["model_id"] = pd.Categorical(agg["model_id"], categories=models, ordered=True)
    agg = agg.sort_values(["model_id", "group"])  # stable order
    # Try a consistent group order if the 4 expected groups are present
    groups = ["B_W", "B_M", "W_W", "W_M"]
    present_groups = [g for g in groups if g in set(agg["group"].unique())]
    hue_order = present_groups if len(present_groups) >= 2 else sorted(agg["group"].unique())
    fig, ax = plt.subplots(figsize=(18, 7))
    sns.barplot(data=agg, x="model_short", y="rate", hue="group", hue_order=hue_order, ax=ax)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Model")
    ax.set_ylabel("Selection rate")
    ax.set_title(f"Selection Rate by Demographic Group (No-abstain) — {dataset_label}")
    add_ideal_fairness_baseline(ax)
    # Increase spacing between model groups by shrinking bar widths and adding margins
    for rect in ax.patches:
        if hasattr(rect, "get_width") and hasattr(rect, "set_width"):
            try:
                rect.set_width(rect.get_width() * 0.7)
            except Exception:
                pass
    ax.margins(x=0.06)
    # Add error bars per bar using precomputed binomial SE
    xticks = [t.get_text() for t in ax.get_xticklabels()]
    bar_idx = 0
    for model_short in xticks:
        for grp in hue_order:
            row = agg[(agg["model_short"] == model_short) & (agg["group"] == grp)].head(1)
            if not row.empty:
                se = float(row["se"].iloc[0]) if not pd.isna(row["se"].iloc[0]) else np.nan
                if not np.isnan(se) and bar_idx < len(ax.patches):
                    bar = ax.patches[bar_idx]
                    x = bar.get_x() + bar.get_width() / 2
                    y = bar.get_height()
                    ax.errorbar(x, y, yerr=se, ecolor="#333333", capsize=3, fmt='none', lw=1)
            bar_idx += 1
    leg = ax.legend(title="Demographic group", loc="upper right", framealpha=0.4)
    if leg and leg.get_frame():
        leg.get_frame().set_alpha(0.4)
    # keep horizontal labels unrotated
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[INFO] Wrote plot -> {out_path}")
    # Debug: verify per-model sum of rates ≈ 2
    try:
        sums = agg.groupby("model_id")["rate"].sum().to_dict()
        counts = agg.groupby("model_id")["n"].max().to_dict()
        print("[DEBUG] selection-rate sum per model (should be 2):", {k: round(v, 4) for k, v in sums.items()})
        print("[DEBUG] mixed pairs per model:", counts)
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Plot selection rates by demographic group from no-abstain fairness runs")
    p.add_argument("--eval_dir", type=str, default="/home/zs7353/resume_validity/evaluations/baseline_noabstain", help="Directory containing no-abstain evaluation CSVs")
    p.add_argument("--dataset", type=str, required=True, choices=["claude", "gemini"], help="Dataset label to filter files")
    p.add_argument("--out", type=str, required=True, help="Output PNG path for the bar plot")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        rows = load_noab_rows(Path(args.eval_dir), args.dataset)
        if rows.empty:
            print("[FATAL] No rows found for the specified dataset; aborting.")
            sys.exit(1)
        agg = compute_selection_rates(rows)
        plot_selection_rates(agg, Path(args.out), dataset_label=args.dataset.capitalize())
    except Exception:
        print("[FATAL] Unhandled exception in no-abstain selection-rate plotting")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()


