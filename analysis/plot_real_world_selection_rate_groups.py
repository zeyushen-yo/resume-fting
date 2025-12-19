#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from plot_noncs_job_metrics import short_model_name, set_style


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Plot real-world selection-rate by gender/race (symmetrized), one row per job")
    p.add_argument("--gender_csv", type=str, required=True)
    p.add_argument("--race_csv", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="analysis/figures_real_world")
    return p.parse_args()


def job_label(df: pd.DataFrame) -> pd.Series:
    comp = df["job_company"].astype(str).str.strip()
    title = df["job_title_posting"].astype(str).str.strip()
    lbl = np.where((comp != "") & (title != ""), comp + " — " + title, np.where(title != "", title, comp))
    return pd.Series(lbl)


def prep(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["job_label"] = job_label(d)
    d["model_short"] = d["model_id"].astype(str).map(short_model_name)
    d["se"] = np.where(d["den"] > 0, np.sqrt(d["selection_rate"] * (1 - d["selection_rate"]) / d["den"]), np.nan)
    return d


def plot_groups(df: pd.DataFrame, title: str, outfile: Path, group_order: List[str]) -> None:
    if df.empty:
        return
    set_style()
    jobs = sorted(df["job_label"].unique().tolist())
    models = sorted(df["model_short"].unique().tolist())
    fig_w = max(18, 2.5 * len(models))
    fig_h = max(10, 4 * len(jobs))
    fig, axes = plt.subplots(len(jobs), 1, figsize=(fig_w, fig_h), squeeze=False)
    for idx, job in enumerate(jobs):
        ax = axes[idx, 0]
        dj = df[df["job_label"] == job].copy()
        if dj.empty:
            ax.axis("off")
            continue
        # Ensure both groups exist per model; pivot helps align
        piv = dj.pivot_table(index=["model_short"], columns="group", values=["selection_rate", "se"])
        # preserve model order
        for i, ms in enumerate(models):
            if ms not in piv.index:
                continue
            x = i
            # Two bars: group_order[0], group_order[1]
            w = 0.35
            for j, grp in enumerate(group_order):
                try:
                    val = float(piv["selection_rate"].loc[ms, grp])
                    se = float(piv["se"].loc[ms, grp])
                except Exception:
                    val = np.nan
                    se = np.nan
                if not np.isnan(val):
                    ax.bar(x + (j - 0.5) * w, val, width=w, label=grp if (idx == 0 and i == 0 and j == 0) else None)
                    if not np.isnan(se):
                        ax.errorbar(x + (j - 0.5) * w, val, yerr=se, ecolor="#333333", capsize=3, fmt="none", lw=1)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=0, ha="center")
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Model")
        ax.set_ylabel("Selection Rate")
        ax.set_title(job)
        ax.legend(title="Group", loc="upper right", framealpha=0.4)
        # baseline
        ax.axhline(0.5, ls="--", lw=1.2, color="#666666")
        ax.text(0.99, 0.51, "random baseline (1/2)", color="#666666", fontsize=10, ha="right", va="bottom", transform=ax.get_yaxis_transform())
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    gender = prep(pd.read_csv(args.gender_csv))
    race = prep(pd.read_csv(args.race_csv))
    out_dir = Path(args.out_dir)
    plot_groups(gender, "Selection Rate by Gender — Real-World Jobs", out_dir / "selection_rate_gender_real_world.png", ["Men", "Women"])
    plot_groups(race, "Selection Rate by Race — Real-World Jobs", out_dir / "selection_rate_race_real_world.png", ["Black", "White"])


if __name__ == "__main__":
    main()



