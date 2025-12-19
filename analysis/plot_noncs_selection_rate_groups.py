#!/usr/bin/env python3
"""
Plot selection-rate disaggregations (gender, race) for non-CS jobs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from plot_noncs_job_metrics import short_model_name, set_style  # reuse helpers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Plot selection-rate by demographic group for non-CS jobs")
    p.add_argument("--gender_csv", type=str, required=True)
    p.add_argument("--race_csv", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="analysis/figures_noncs")
    return p.parse_args()


def prep_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["model_short"] = df["model_id"].astype(str).map(short_model_name)
    df["se"] = np.where(
        df["den"] > 0,
        np.sqrt(df["selection_rate"] * (1.0 - df["selection_rate"]) / df["den"]),
        np.nan,
    )
    return df


def bars_by_job_models(
    df: pd.DataFrame,
    group_order: List[str],
    title: str,
    outfile: Path,
    ylabel: str,
) -> None:
    if df.empty:
        return
    jobs = sorted(df["job_type"].unique().tolist())
    models = sorted(df["model_short"].unique().tolist())
    set_style()
    fig_w = max(18, 2.5 * len(models))
    fig_h = max(10, 4 * len(jobs))
    fig, axes = plt.subplots(len(jobs), 1, figsize=(fig_w, fig_h), squeeze=False)
    for idx, job in enumerate(jobs):
        ax = axes[idx, 0]
        dj = df[df["job_type"] == job]
        if dj.empty:
            ax.axis("off")
            continue
        sns.barplot(
            data=dj,
            x="model_short",
            y="selection_rate",
            hue="group",
            hue_order=group_order,
            order=models,
            ax=ax,
            palette="Set2",
            errorbar=None,
        )
        patches = ax.patches
        for patch, (_, row) in zip(patches, dj.sort_values(["model_short", "group"]).iterrows()):
            se_val = row["se"]
            if not pd.isna(se_val):
                x = patch.get_x() + patch.get_width() / 2
                y = patch.get_height()
                ax.errorbar(x, y, yerr=se_val, ecolor="#333333", capsize=3, fmt="none", lw=1)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Model")
        ax.set_ylabel(ylabel)
        ax.set_title(job)
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        ax.legend(title="Group", loc="upper right", framealpha=0.4)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    gender_df = prep_df(Path(args.gender_csv))
    race_df = prep_df(Path(args.race_csv))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bars_by_job_models(
        gender_df,
        group_order=["Men", "Women"],
        title="Selection Rate by Gender — Non-CS Jobs",
        outfile=out_dir / "selection_rate_gender_noncs_jobs.png",
        ylabel="Selection Rate",
    )
    bars_by_job_models(
        race_df,
        group_order=["Black", "White"],
        title="Selection Rate by Race — Non-CS Jobs",
        outfile=out_dir / "selection_rate_race_noncs_jobs.png",
        ylabel="Selection Rate",
    )


if __name__ == "__main__":
    main()



