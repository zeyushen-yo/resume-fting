#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from plot_noncs_job_metrics import short_model_name, set_style, plot_metric  # reuse style helpers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Plot real-world metrics by job (one row per posting), models on x-axis")
    p.add_argument("--summary_csv", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="analysis/figures_real_world")
    p.add_argument("--pairs_root", type=str, default="/home/zs7353/resume_validity/data/pairs_from_real_world")
    return p.parse_args()


def _combine_job_label(row: pd.Series) -> str:
    company = str(row.get("job_company") or "").strip()
    title = str(row.get("job_title_posting") or "").strip()
    if company and title:
        return f"{company} — {title}"
    return title or company or "Unknown job"


def _compute_se(df: pd.DataFrame, metric_col: str) -> pd.Series:
    if metric_col in ("criterion_validity", "unjustified_abstention"):
        denom = df["n_strict_pairs"].astype(float)
    elif metric_col == "discriminant_validity":
        denom = df["n_equal_pairs"].astype(float)
    else:
        denom = df["n_equal_pairs_noab"].astype(float)
    p = df[metric_col].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.sqrt(p * (1.0 - p) / denom)
    se[~np.isfinite(se)] = np.nan
    return se


def add_baseline(ax, value: float, label: str) -> None:
    ax.axhline(value, ls="--", lw=1.2, color="#666666")
    ax.text(
        0.99,
        value + 0.01,
        label,
        color="#666666",
        fontsize=10,
        ha="right",
        va="bottom",
        transform=ax.get_yaxis_transform(),
    )


def bars_by_job(df: pd.DataFrame, metric_col: str, title: str, out_path: Path, baseline: float | None = None, baseline_label: str | None = None) -> None:
    if df.empty:
        return
    set_style()
    df = df.copy()
    df["job_label"] = df.apply(_combine_job_label, axis=1)
    df["model_short"] = df["model_id"].astype(str).map(short_model_name)
    jobs: List[str] = sorted(df["job_label"].unique().tolist())
    models: List[str] = sorted(df["model_short"].unique().tolist())

    fig_w = max(18, 2.5 * len(models))
    fig_h = max(10, 4 * len(jobs))
    fig, axes = plt.subplots(len(jobs), 1, figsize=(fig_w, fig_h), squeeze=False)

    for idx, job in enumerate(jobs):
        ax = axes[idx, 0]
        dj = df[df["job_label"] == job].copy().sort_values("model_short")
        if dj.empty:
            ax.axis("off")
            continue
        dj["se"] = _compute_se(dj, metric_col)
        sns.barplot(data=dj, x="model_short", y=metric_col, ax=ax, order=models, errorbar=None, palette="Set2")
        # error bars
        for bar_idx, ms in enumerate(models):
            row = dj[dj["model_short"] == ms].head(1)
            if not row.empty and pd.notna(row["se"].iloc[0]):
                if bar_idx < len(ax.patches):
                    bar = ax.patches[bar_idx]
                    x = bar.get_x() + bar.get_width() / 2
                    y = bar.get_height()
                    ax.errorbar(x, y, yerr=float(row["se"].iloc[0]), ecolor="#333333", capsize=3, fmt="none", lw=1)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(job)
        ax.set_xlabel("Model")
        ax.set_ylabel(metric_col.replace("_", " ").title())
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        if baseline is not None and baseline_label is not None:
            add_baseline(ax, baseline, baseline_label)

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    src = Path(args.summary_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src)
    if df.empty:
        return
    # Restrict to jobs present under pairs_root to avoid mixing with other runs
    try:
        import json
        allowed: set[tuple[str, str]] = set()
        pairs_root = Path(args.pairs_root)
        for pf in pairs_root.rglob("pairs_shard_00_of_01.jsonl"):
            with open(pf, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    js = obj.get("job_source") or {}
                    comp = str(js.get("company") or "").strip()
                    title = str(js.get("title") or "").strip()
                    allowed.add((comp, title))
                    break  # only need first line
        if allowed:
            df = df[df.apply(lambda r: (str(r.get("job_company") or "").strip(), str(r.get("job_title_posting") or "").strip()) in allowed, axis=1)]
    except Exception:
        pass
    # Reuse non-CS plotting helpers for identical style
    df_plot = df.copy()
    df_plot["job_type"] = df_plot.apply(_combine_job_label, axis=1)
    df_plot["model_short"] = df_plot["model_id"].astype(str).map(short_model_name)
    set_style()
    # criterion validity
    plot_metric(
        df_plot,
        value_col="criterion_validity",
        denom_col="n_strict_pairs",
        ylabel="Criterion Validity",
        title="Criterion Validity — Real-World Jobs",
        outfile=out_dir / "criterion_validity_real_world.png",
        baseline=1/3,
        baseline_label="random baseline (1/3)",
    )
    # unjustified abstention
    plot_metric(
        df_plot,
        value_col="unjustified_abstention",
        denom_col="n_strict_pairs",
        ylabel="Unjustified Abstention",
        title="Unjustified Abstention — Real-World Jobs",
        outfile=out_dir / "unjustified_abstention_real_world.png",
    )
    # discriminant validity
    plot_metric(
        df_plot,
        value_col="discriminant_validity",
        denom_col="n_equal_pairs",
        ylabel="Discriminant Validity",
        title="Discriminant Validity — Real-World Jobs",
        outfile=out_dir / "discriminant_validity_real_world.png",
    )
    # selection rate (first)
    plot_metric(
        df_plot,
        value_col="selection_rate_first",
        denom_col="n_equal_pairs_noab",
        ylabel="Selection Rate (First Resume)",
        title="Selection Rate on Equal Pairs — Real-World Jobs",
        outfile=out_dir / "selection_rate_real_world.png",
        baseline=0.5,
        baseline_label="random baseline (1/2)",
    )


if __name__ == "__main__":
    main()


