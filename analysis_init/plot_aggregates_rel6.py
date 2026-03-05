#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple
import re

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
import time


def safe_name(s: str) -> str:
    return (
        s.replace("/", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace(".", "_")
        .replace("-", "-")
    )


def read_json_df(path: Path) -> pd.DataFrame:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except Exception:
        print(f"[ERROR] Failed reading {path}")
        print(traceback.format_exc())
        return pd.DataFrame()


def ensure_cols(df: pd.DataFrame, cols: List[Tuple[str, object]]) -> pd.DataFrame:
    for c, default in cols:
        if c not in df.columns:
            df[c] = default
    return df


def normalize_qual_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Remove common bullets/markdown and excessive whitespace
    t = text.replace("\t", " ")
    t = re.sub(r"^[\s•\-*+>]+", "", t)  # leading bullets
    t = re.sub(r"\s{2,}", " ", t)
    t = t.strip().strip("-•*+·:")
    # Remove orphan markdown remnants like '**' at ends
    t = t.replace("**", "")
    return t.strip()


def list_csv_files(root: Path, includes: List[str] | None = None, excludes: List[str] | None = None) -> List[Path]:
    includes = includes or []
    excludes = excludes or []
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


def load_baseline_df(baseline_dir: Path, includes: List[str], excludes: List[str]) -> pd.DataFrame:
    paths = list_csv_files(baseline_dir, includes, excludes)
    rows: List[pd.DataFrame] = []
    for fp in paths:
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        model_id = infer_model_from_path(fp)
        # Force shorthand model IDs everywhere in baseline rows
        short_id = short_model_name(model_id)
        df["model_id"] = short_id
        df["model_short"] = short_id
        # Normalize
        if "experiment_type_norm" in df.columns:
            et = df["experiment_type_norm"].astype(str)
        elif "experiment_type" in df.columns:
            et = df["experiment_type"].astype(str)
        else:
            et = pd.Series([""] * len(df))
        df["experiment_type"] = et.str.lower()
        df["pair_type"] = df.get("pair_type", "").astype(str).str.lower()
        df["decision"] = df.get("decision", "").astype(str).str.lower()
        # abstained to bool
        df["abstained"] = df.get("abstained", False)
        if df["abstained"].dtype != bool:
            df["abstained"] = df["abstained"].astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])
        # demographics
        df["demographic_base"] = df.get("demographic_base", "").astype(str)
        df["demographic_variant"] = df.get("demographic_variant", "").astype(str)
        df["better"] = df.get("better", "").astype(str).str.lower()
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def eval_file_filters_for_dataset(label: str) -> tuple[list[str], list[str]]:
    """Return include/exclude substrings to select evaluation CSVs for a dataset label.
    - For 'claude': match only files with _claude_rel6_shard
    - For 'gemini': match generic _rel6_shard but exclude _claude_rel6_shard
    """
    lab = (label or "").strip().lower()
    if lab == "claude":
        return ["_claude_rel6_shard"], []
    if lab == "gemini":
        return ["_rel6_shard"], ["_claude_rel6_shard"]
    return ["_rel6_shard"], []


def weighted_group_mean(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    # expects columns: accuracy (float), n (int)
    if df.empty:
        return df
    df = df.copy()
    df["n"] = pd.to_numeric(df["n"], errors="coerce").fillna(0).astype(int)
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce").fillna(0.0)
    df["acc_n"] = df["accuracy"] * df["n"]
    g = df.groupby(group_cols, dropna=False, as_index=False).agg({"acc_n": "sum", "n": "sum"})
    g["accuracy"] = np.where(g["n"] > 0, g["acc_n"] / g["n"], np.nan)
    return g.drop(columns=["acc_n"])  # keep accuracy, n + group_cols


def set_style():
    # Cohesive, readable style
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


def add_random_baseline(ax, y: float = 0.5, label: str = "random baseline") -> None:
    ax.axhline(y, ls="--", lw=1.2, color="#666666")
    ax.text(0.99, y + 0.01, label, color="#666666", fontsize=10, ha="right", va="bottom", transform=ax.get_yaxis_transform())


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


def plot_validity_vs_k_grid(df_pair_num: pd.DataFrame, models: List[str], out_path: Path, dataset_label: str) -> None:
    if df_pair_num.empty:
        return
    # Weighted average over pair_type for each (model_id, num_differed), plus simple SE for error bars
    df = df_pair_num.copy()
    df = df[df["num_differed"].isin([1, 2, 3])]
    # Compute accuracy as weighted mean across pair_types; estimate variance by binomial with n
    dfw = weighted_group_mean(df, ["model_id", "num_differed"])  # yields accuracy, n
    dfw["se"] = np.where(dfw["n"] > 0, np.sqrt(dfw["accuracy"] * (1 - dfw["accuracy"]) / dfw["n"]), np.nan)

    # 3x3 subplots
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    for idx, m in enumerate(models[:9]):
        ax = axes[idx]
        sdf = dfw[dfw["model_id"] == m].sort_values("num_differed")
        if sdf.empty:
            ax.axis('off')
            continue
        ax.errorbar(sdf["num_differed"], sdf["accuracy"], yerr=sdf["se"], fmt='-o', color="#266DD3", ecolor="#7EA8F8", capsize=3)
        ax.set_title(short_model_name(m))
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("k")
        ax.set_ylabel("Validity")
        # Random baseline for validity plots is 1/3
        add_random_baseline(ax, y=1/3, label="random baseline (1/3)")
        # integer ticks at 1,2,3
        ax.set_xticks([1, 2, 3])
    # Remove unused axes if <9 models
    for j in range(len(models), 9):
        axes[j].axis('off')
    fig.suptitle(f"Validity vs k (1..3) — {dataset_label}", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)


def compute_validity_by_num_from_baseline_same_demo(baseline_dir: Path, dataset_label: str) -> pd.DataFrame:
    includes, excludes = eval_file_filters_for_dataset(dataset_label)
    rows = load_baseline_df(baseline_dir, includes=includes, excludes=excludes)
    if rows.empty:
        return pd.DataFrame()
    # Only validity-style experiments
    rows = rows[rows["experiment_type"].isin(["validity", "validity_demographics"])].copy()
    # Only same-demographic pairs among canonical groups
    allowed = {"B_W", "B_M", "W_W", "W_M"}
    rows = rows[(rows["demographic_base"].astype(str) == rows["demographic_variant"].astype(str)) &
                (rows["demographic_base"].isin(allowed))]
    if rows.empty:
        return pd.DataFrame()
    # Normalize decisions; abstention counts as incorrect
    dec = rows["decision"].astype(str).str.lower()
    dec = dec.where(~rows["abstained"].astype(bool), other="abstain")
    gt = rows["better"].astype(str).str.lower()
    rows["is_correct"] = ((dec == "first") & (gt == "first")) | ((dec == "second") & (gt == "second"))
    # num_differed must be present
    if "num_differed" not in rows.columns:
        return pd.DataFrame()
    try:
        rows["num_differed"] = pd.to_numeric(rows["num_differed"], errors="coerce").astype("Int64")
    except Exception:
        return pd.DataFrame()
    g = rows.groupby(["model_id", "num_differed"], dropna=False).agg(n=("is_correct", "size"), k=("is_correct", "sum")).reset_index()
    g["accuracy"] = np.where(g["n"] > 0, g["k"] / g["n"], np.nan)
    return g[["model_id", "num_differed", "accuracy", "n"]]


def plot_validity_vs_k_grid_combined(
    df_pair_num_a: pd.DataFrame,
    df_pair_num_b: pd.DataFrame,
    models: List[str],
    out_path: Path,
    label_a: str,
    label_b: str,
) -> None:
    if df_pair_num_a.empty or df_pair_num_b.empty:
        return
    def prep(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df[df["num_differed"].isin([1, 2, 3])]
        d = weighted_group_mean(df, ["model_id", "num_differed"])  # accuracy, n
        d["se"] = np.where(d["n"] > 0, np.sqrt(d["accuracy"] * (1 - d["accuracy"]) / d["n"]), np.nan)
        return d
    a = prep(df_pair_num_a)
    b = prep(df_pair_num_b)
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    for idx, m in enumerate(models[:9]):
        ax = axes[idx]
        sa = a[a["model_id"] == m].sort_values("num_differed")
        sb = b[b["model_id"] == m].sort_values("num_differed")
        if sa.empty and sb.empty:
            ax.axis('off')
            continue
        if not sa.empty:
            ax.errorbar(sa["num_differed"], sa["accuracy"], yerr=sa["se"], fmt='-o', color="#266DD3", ecolor="#7EA8F8", capsize=3, label=label_a)
        if not sb.empty:
            ax.errorbar(sb["num_differed"], sb["accuracy"], yerr=sb["se"], fmt='-o', color="#E66A00", ecolor="#F4B183", capsize=3, label=label_b)
        ax.set_title(short_model_name(m))
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("k")
        ax.set_ylabel("Validity")
        add_random_baseline(ax, y=1/3, label="random baseline (1/3)")
        ax.legend(loc="lower left", fontsize=9, framealpha=0.6)
        ax.set_xticks([1, 2, 3])
    for j in range(len(models), 9):
        axes[j].axis('off')
    fig.suptitle("Validity vs k (1..3) — Combined (Claude vs Gemini)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)


def plot_pair_type_bars(df_pair_num: pd.DataFrame, models: List[str], out_path: Path, dataset_label: str) -> None:
    if df_pair_num.empty:
        return
    # Weighted average over num_differed: per (model_id, pair_type)
    df = weighted_group_mean(df_pair_num, ["model_id", "pair_type"])  # accuracy, n
    df["se"] = np.where(df["n"] > 0, np.sqrt(df["accuracy"] * (1 - df["accuracy"]) / df["n"]), np.nan)
    df = df[df["pair_type"].isin(["preferred", "reworded", "underqualified"])].copy()
    df["model_id"] = pd.Categorical(df["model_id"], categories=models, ordered=True)
    df["model_short"] = df["model_id"].astype(str).map(short_model_name)
    df = df.sort_values(["model_id", "pair_type"])  # stable order
    fig, ax = plt.subplots(figsize=(16, 7))
    palette = sns.color_palette("Set2", n_colors=3)
    hue_order = ["preferred", "reworded", "underqualified"]
    sns.barplot(data=df, x="model_short", y="accuracy", hue="pair_type", hue_order=hue_order, ax=ax, palette=palette, errorbar=None)
    # Add error bars manually matching seaborn bar order
    xticks = [t.get_text() for t in ax.get_xticklabels()]
    bar_idx = 0
    for model_short in xticks:
        for pt in hue_order:
            row = df[(df["model_short"] == model_short) & (df["pair_type"] == pt)].head(1)
            if not row.empty:
                se = float(row["se"].iloc[0]) if not pd.isna(row["se"].iloc[0]) else np.nan
                if not np.isnan(se):
                    bar = ax.patches[bar_idx]
                    x = bar.get_x() + bar.get_width() / 2
                    y = bar.get_height()
                    ax.errorbar(x, y, yerr=se, ecolor="#333333", capsize=3, fmt='none', lw=1)
            bar_idx += 1
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Model")
    ax.set_ylabel("Validity")
    ax.set_title(f"Validity by Pair Type — {dataset_label}")
    # Random baseline for validity plots is 1/3
    add_random_baseline(ax, y=1/3, label="random baseline (1/3)")
    # shrink bar widths and add x margin to reduce label clash (robust)
    for rect in ax.patches:
        if hasattr(rect, "get_width") and hasattr(rect, "set_width"):
            w = rect.get_width()
            try:
                rect.set_width(w * 0.8)
            except Exception:
                pass
    ax.margins(x=0.06)
    # Legend: inside top-right with transparency to reduce clutter
    leg = ax.legend(title="Pair Type", loc="upper right", framealpha=0.4)
    if leg and leg.get_frame():
        leg.get_frame().set_alpha(0.4)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_pair_type_bars_noabstain(models: List[str], out_path: Path, dataset_label: str, agg_dir: Path) -> None:
    # Compute conditional validity (given non-abstention) from baseline rows
    base_dir = agg_dir.parents[2] / "evaluations" / "baseline"
    includes, excludes = eval_file_filters_for_dataset(dataset_label)
    rows = load_baseline_df(base_dir, includes=includes, excludes=excludes)
    if rows.empty:
        return
    rows = rows[rows["experiment_type"].isin(["validity", "validity_demographics"])].copy()
    rows = rows[rows["pair_type"].isin(["preferred", "underqualified"])].copy()
    # Only non-abstained rows, and only cases with a defined better candidate (first/second)
    dec = rows["decision"].astype(str).str.lower()
    rows["non_abstain"] = ~rows["abstained"].astype(bool)
    gt = rows["better"].astype(str).str.lower()
    mask_eval = rows["non_abstain"] & gt.isin(["first", "second"]) & dec.isin(["first", "second"]) 
    sub = rows[mask_eval].copy()
    if sub.empty:
        return
    sub["is_correct"] = ((dec == "first") & (gt == "first")) | ((dec == "second") & (gt == "second"))
    g = sub.groupby(["model_id", "pair_type"]).agg(n=("is_correct", "size"), k=("is_correct", "sum")).reset_index()
    g["rate"] = np.where(g["n"] > 0, g["k"] / g["n"], np.nan)
    g["se"] = np.sqrt(g["rate"] * (1 - g["rate"]) / g["n"].replace(0, np.nan))
    g["model_short"] = g["model_id"].map(short_model_name)
    # Plot
    fig, ax = plt.subplots(figsize=(16, 7))
    sns.barplot(data=g, x="model_short", y="rate", hue="pair_type", ax=ax, errorbar=None)
    # error bars
    xticks = [t.get_text() for t in ax.get_xticklabels()]
    hue_vals = sorted(g["pair_type"].unique())
    bar_idx = 0
    for ms in xticks:
        for hv in hue_vals:
            row = g[(g["model_short"] == ms) & (g["pair_type"] == hv)].head(1)
            if not row.empty:
                se = float(row["se"].iloc[0]) if not pd.isna(row["se"].iloc[0]) else np.nan
                if not np.isnan(se) and bar_idx < len(ax.patches):
                    bar = ax.patches[bar_idx]
                    x = bar.get_x() + bar.get_width() / 2
                    y = bar.get_height()
                    ax.errorbar(x, y, yerr=se, ecolor="#333333", capsize=3, fmt='none', lw=1)
            bar_idx += 1
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Model")
    ax.set_ylabel("Conditional Validity (non-abstain)")
    ax.set_title(f"Validity (Non-abstain): Preferred vs Underqualified ({dataset_label})")
    # Random baseline among two choices
    add_random_baseline(ax, y=0.5, label="random baseline (1/2)")
    # spacing
    for rect in ax.patches:
        if hasattr(rect, "get_width") and hasattr(rect, "set_width"):
            try:
                rect.set_width(rect.get_width() * 0.8)
            except Exception:
                pass
    ax.margins(x=0.06)
    leg = ax.legend(title="Pair Type", loc="upper right", framealpha=0.4)
    if leg and leg.get_frame():
        leg.get_frame().set_alpha(0.4)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_heatmap(
    df: pd.DataFrame,
    index_col: str,
    columns_col: str,
    value_col: str,
    out_path: Path,
    title: str,
    cmap: str = "RdPu",
    row_scale: float = 0.4,
    min_height: float = 6.0,
    title_fontsize: int | None = None,
    title_pad: float | None = None,
    xtick_rotation: int | None = None,
    ytick_rotation: int | None = None,
    use_square_cells: bool = False,
    cell_size: float = 0.7,
    fig_w: float | None = None,
    fig_h: float | None = None,
    bottom_margin: float | None = None,
    use_suptitle: bool = False,
    suptitle_fontsize: int | None = None,
    suptitle_y: float | None = None,
    tight_rect: tuple[float, float, float, float] | None = None,
    xlabel_fontsize: int | None = None,
    ylabel_fontsize: int | None = None,
    tick_labelsize: int | None = None,
    cbar_tick_labelsize: int | None = None,
) -> None:
    if df.empty:
        return
    # Pivot and order
    p = df.pivot_table(index=index_col, columns=columns_col, values=value_col, aggfunc="mean")
    p = p.sort_index()
    # Standardize orientation: models on rows, items (jobs/demos/clusters) on columns
    if index_col != "model_short" and columns_col == "model_short":
        p = p.T
    if fig_w is None or fig_h is None:
        if use_square_cells:
            fig_w = max(10, cell_size * p.shape[1] + 3)
            fig_h = max(10, cell_size * p.shape[0] + 3)
        else:
            fig_w = max(10, 0.6 * p.shape[1] + 3)
            fig_h = max(min_height, row_scale * p.shape[0] + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    hm = sns.heatmap(p, ax=ax, cmap=cmap, vmin=0.0, vmax=1.0, cbar_kws={"label": "Validity", "shrink": 1.0}, square=use_square_cells)
    if use_suptitle:
        kwargs = {"fontweight": "bold"}
        if suptitle_fontsize is not None:
            kwargs["fontsize"] = suptitle_fontsize
        if suptitle_y is not None:
            kwargs["y"] = suptitle_y
        fig.suptitle(title, **kwargs)
    else:
        if title_fontsize is not None:
            if title_pad is not None:
                ax.set_title(title, fontweight="bold", fontsize=title_fontsize, pad=title_pad)
            else:
                ax.set_title(title, fontweight="bold", fontsize=title_fontsize)
        else:
            if title_pad is not None:
                ax.set_title(title, fontweight="bold", pad=title_pad)
            else:
                ax.set_title(title, fontweight="bold")
    xlab = columns_col.replace("model_short", "model")
    ylab = index_col.replace("model_short", "model")
    if xlabel_fontsize is not None:
        ax.set_xlabel(xlab, fontweight="bold", fontsize=xlabel_fontsize)
    else:
        ax.set_xlabel(xlab, fontweight="bold")
    if ylabel_fontsize is not None:
        ax.set_ylabel(ylab, fontweight="bold", fontsize=ylabel_fontsize)
    else:
        ax.set_ylabel(ylab, fontweight="bold")
    # Control tick label rotation/orientation as requested
    if xtick_rotation is not None:
        plt.setp(ax.get_xticklabels(), rotation=xtick_rotation, ha="right" if xtick_rotation else "center")
    if ytick_rotation is not None:
        plt.setp(ax.get_yticklabels(), rotation=ytick_rotation, va="center")
    if tick_labelsize is not None:
        ax.tick_params(axis="both", labelsize=tick_labelsize)
    # Colorbar tick and label sizes
    try:
        cbar = hm.collections[0].colorbar
        if cbar is not None:
            if cbar_tick_labelsize is not None:
                cbar.ax.tick_params(labelsize=cbar_tick_labelsize)
            if xlabel_fontsize is not None:
                cbar.ax.yaxis.label.set_size(xlabel_fontsize)
    except Exception:
        pass
    if bottom_margin is not None:
        try:
            fig.subplots_adjust(bottom=bottom_margin)
        except Exception:
            pass
    if bottom_margin is not None:
        try:
            fig.subplots_adjust(bottom=bottom_margin)
        except Exception:
            pass
    if tight_rect is not None:
        fig.tight_layout(rect=tight_rect)
    else:
        fig.tight_layout()
    
    # Adjust colorbar height to match heatmap AFTER tight_layout
    try:
        cbar = hm.collections[0].colorbar
        if cbar is not None:
            cbar.ax.set_position([cbar.ax.get_position().x0, ax.get_position().y0, 
                                 cbar.ax.get_position().width, ax.get_position().height])
    except Exception:
        pass
    
    fig.savefig(out_path)
    plt.close(fig)


def build_demo_combo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["demographic_base"] = df["demographic_base"].astype(str)
    df["demographic_variant"] = df["demographic_variant"].astype(str)
    # Use slash instead of arrow for clarity and to avoid directional implication
    df["demo_combo"] = df["demographic_base"] + " / " + df["demographic_variant"]
    return df


def plot_valid_vs_valid_demo(df_valid_overall: pd.DataFrame, df_valid_demo_overall: pd.DataFrame, models: List[str], out_path: Path, dataset_label: str) -> None:
    if df_valid_overall.empty and df_valid_demo_overall.empty:
        return
    left = df_valid_overall[["model_id", "accuracy", "n"]].copy()
    left["se_v"] = np.where(left["n"] > 0, np.sqrt(left["accuracy"] * (1 - left["accuracy"]) / left["n"]), np.nan)
    left = left.rename(columns={"accuracy": "validity"})
    right = df_valid_demo_overall[["model_id", "accuracy", "n"]].copy()
    right["se_vd"] = np.where(right["n"] > 0, np.sqrt(right["accuracy"] * (1 - right["accuracy"]) / right["n"]), np.nan)
    right = right.rename(columns={"accuracy": "validity_demographics"})
    df = pd.merge(left, right, on="model_id", how="outer")
    df["model_id"] = pd.Categorical(df["model_id"], categories=models, ordered=True)
    df["model_short"] = df["model_id"].astype(str).map(short_model_name)
    df_melt = df.melt(id_vars=["model_id", "model_short", "se_v", "se_vd"], value_vars=["validity", "validity_demographics"], var_name="experiment", value_name="accuracy")
    fig, ax = plt.subplots(figsize=(16, 7))
    palette = {"validity": "#1f77b4", "validity_demographics": "#ff7f0e"}
    hue_order = ["validity", "validity_demographics"]
    sns.barplot(data=df_melt, x="model_short", y="accuracy", hue="experiment", hue_order=hue_order, ax=ax, palette=palette, errorbar=None)
    # Random baseline for validity plots is 1/3
    add_random_baseline(ax, y=1/3, label="random baseline (1/3)")
    # shrink bar widths and add margin for spacing similar to other bar plots (robust)
    for rect in ax.patches:
        if hasattr(rect, "get_width") and hasattr(rect, "set_width"):
            w = rect.get_width()
            try:
                rect.set_width(w * 0.8)
            except Exception:
                pass
    ax.margins(x=0.06)
    # Add error bars with correct mapping
    xticks = [t.get_text() for t in ax.get_xticklabels()]
    bar_idx = 0
    for model_short in xticks:
        for exp in hue_order:
            if exp == "validity":
                se = float(df[df["model_short"] == model_short]["se_v"].iloc[0]) if not df[df["model_short"] == model_short].empty else np.nan
            else:
                se = float(df[df["model_short"] == model_short]["se_vd"].iloc[0]) if not df[df["model_short"] == model_short].empty else np.nan
            if not np.isnan(se):
                bar = ax.patches[bar_idx]
                x = bar.get_x() + bar.get_width() / 2
                y = bar.get_height()
                ax.errorbar(x, y, yerr=se, ecolor="#333333", capsize=3, fmt='none', lw=1)
            bar_idx += 1
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Model")
    ax.set_ylabel("Validity")
    ax.set_title(f"Validity: Validity vs Validity_Demographics — {dataset_label}")
    # Legend inside with slight transparency
    # Align legend transparency with other plots (0.4)
    leg = ax.legend(title="Experiment", loc="upper right", framealpha=0.4)
    if leg and leg.get_frame():
        leg.get_frame().set_alpha(0.4)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def openrouter_gemini_summarize(bullets: List[str], api_key: str, retries: int = 4) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    system = (
        "You are an expert HR analyst. Given a list of job qualification bullet points, "
        "produce a short human-readable cluster label (max 6 words) describing the theme. "
        "Do not include punctuation like quotes or trailing periods."
    )
    user = "\n".join(f"- {b}" for b in bullets)
    payload = {
        "model": "google/gemini-2.5-pro",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": 64,
        "temperature": 0.2,
    }
    for attempt in range(retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"].strip()
            return text
        except Exception as e:
            print(f"[WARN] Gemini summarize failed (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return ""


def cluster_diff_quals_and_plot(
    df_v: pd.DataFrame,
    df_vd: pd.DataFrame,
    models: List[str],
    out_path: Path,
    dataset_label: str,
    n_clusters: int = 20,
    gemini_api_key: str = "",
    exclude_labels: List[str] | None = None,
    reuse_cache_only: bool = False,
    exclude_ids: List[int] | None = None,
) -> None:
    # df_v, df_vd columns: model_id, differed_qualification, n, accuracy
    if df_v.empty and df_vd.empty:
        return
    cols_needed = ["model_id", "differed_qualification", "n", "accuracy"]
    for df in (df_v, df_vd):
        for c in cols_needed:
            if c not in df.columns:
                df[c] = np.nan
    # Combine, but cluster on unique qual strings
    qual_series = pd.concat([
        df_v["differed_qualification"].dropna().astype(str),
        df_vd["differed_qualification"].dropna().astype(str),
    ], ignore_index=True)
    # Clean and filter out very short fragments
    qual_series = qual_series.apply(normalize_qual_text)
    qual_series = qual_series[qual_series.str.len() >= 12]
    qual_series = qual_series[qual_series.str.strip() != ""].drop_duplicates().reset_index(drop=True)
    if qual_series.empty:
        return
    try:
        # Use full qualification text without chunking; allow bigram features but no aggressive tokenization
        vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
        X = vec.fit_transform(qual_series.tolist())
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        qual_to_cluster: Dict[str, int] = {q: int(c) for q, c in zip(qual_series.tolist(), labels)}
        # Summarize clusters using Gemini-2.5-pro (Google API). Support cache reuse.
        cache_dir = Path("/home/zs7353/resume_validity/analysis/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "cluster_summaries_gemini25.json"
        cache: Dict[str, str] = {}
        if cache_path.exists():
            try:
                cache = json.loads(cache_path.read_text())
            except Exception as e:
                print(f"[WARN] Failed to read cache: {e}")
        cluster_names: Dict[int, str] = {}
        if reuse_cache_only:
            # Load all labels from cache; hard-fail if any missing
            for cid in range(n_clusters):
                key = f"gemini25_cluster_{cid}"
                lbl = cache.get(key, "").strip()
                if not lbl:
                    print(f"[FATAL] Missing cached label for cluster {cid}. Reuse mode requires complete cache.")
                    sys.exit(1)
                cluster_names[cid] = lbl
        else:
            api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                print("[FATAL] GEMINI_API_KEY not set; cannot label clusters.")
                sys.exit(1)
            # Otherwise, generate labels and persist to cache
            cluster_names = {}
        # Prepare sample bullets per cluster
        qual_df = pd.DataFrame({"q": qual_series.tolist(), "cluster": labels})
        for cid in range(n_clusters):
            samples = [normalize_qual_text(s) for s in qual_df[qual_df["cluster"] == cid]["q"].head(10).tolist()]
            key = f"gemini25_cluster_{cid}"
            if reuse_cache_only:
                # Already loaded in cluster_names above
                continue
            label = ""
            if api_key:
                # Use Google Generative Language API directly
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key={api_key}"
                    system = (
                        "You are an expert HR analyst. Given a list of job qualification bullet points, "
                        "produce a short, human-readable label (max 6 words) naming the common theme. "
                        "Return ONLY the label text without any quotes, punctuation, numbering, or explanations."
                    )
                    prompt = "\n".join(f"- {b}" for b in samples)
                    payload = {
                        "systemInstruction": {"role": "system", "parts": [{"text": system}]},
                        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8192, "candidateCount": 1},
                    }
                    # Retry a few times with basic backoff
                    for attempt in range(4):
                        resp = requests.post(url, json=payload, timeout=60)
                        if resp.ok:
                            data = resp.json()
                            try:
                                cand = data.get("candidates", [{}])[0]
                                content = cand.get("content", {})
                                parts = content.get("parts", [])
                                text = "".join([p.get("text", "") for p in parts]).strip() if parts else ""
                            except Exception as e:
                                print(f"[WARN] Gemini response parse error for cluster {cid}: {e}")
                                text = ""
                            if text:
                                label = text
                                print(f"[INFO] Cluster {cid} label (Gemini): {label}")
                                break
                            else:
                                # Print truncated response for debugging per user preference
                                try:
                                    dbg = json.dumps(data)[:600]
                                except Exception:
                                    dbg = str(data)[:600]
                                print(f"[WARN] Empty label with 200 OK for cluster {cid}. Body: {dbg}")
                        else:
                            print(f"[WARN] Gemini HTTP {resp.status_code} for cluster {cid}: {resp.text[:400]}")
                        time.sleep(2 ** attempt)
                except Exception as e:
                    print(f"[WARN] Gemini API summarize failed for cluster {cid}: {e}")
            if not label:
                # Fallback path: OpenRouter Gemini-2.5-pro using OPENROUTER_API_KEY
                or_key = os.environ.get("OPENROUTER_API_KEY", "")
                if or_key:
                    try:
                        or_url = "https://openrouter.ai/api/v1/chat/completions"
                        headers = {"Authorization": f"Bearer {or_key}", "Content-Type": "application/json"}
                        payload_or = {
                            "model": "google/gemini-2.5-pro",
                            "messages": [
                                {"role": "system", "content": system},
                                {"role": "user", "content": "\n".join(f"- {b}" for b in samples)},
                            ],
                            "max_tokens": 64,
                            "temperature": 0.2,
                        }
                        for attempt in range(3):
                            rr = requests.post(or_url, headers=headers, json=payload_or, timeout=60)
                            if rr.ok:
                                jd = rr.json()
                                try:
                                    label = jd["choices"][0]["message"]["content"].strip()
                                except Exception as e:
                                    print(f"[WARN] OpenRouter parse error for cluster {cid}: {e}")
                                    label = ""
                                if label:
                                    print(f"[INFO] Cluster {cid} label (OpenRouter Gemini): {label}")
                                    break
                            else:
                                print(f"[WARN] OpenRouter HTTP {rr.status_code} for cluster {cid}: {rr.text[:400]}")
                            time.sleep(2 ** attempt)
                    except Exception as e:
                        print(f"[WARN] OpenRouter Gemini fallback failed for cluster {cid}: {e}")
            if not label:
                print(f"[FATAL] Empty label from Gemini for cluster {cid}; aborting.")
                sys.exit(1)
            # Persist to cache for future reuse
            try:
                cache[key] = label
                cache_path.write_text(json.dumps(cache, indent=2))
            except Exception as e:
                print(f"[WARN] Failed to write cluster label cache: {e}")
            cluster_names[cid] = label
    except Exception:
        print("[ERROR] TF-IDF/KMeans failed; falling back to naive grouping by first letter")
        print(traceback.format_exc())
        # Fallback grouping (rare): 20 buckets by hash
        qual_to_cluster = {q: (hash(q) % n_clusters) for q in qual_series.tolist()}
        cluster_names = {i: f"Cluster {i+1:02d}" for i in range(n_clusters)}

    def assign_cluster(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df["cluster_id"] = df["differed_qualification"].map(lambda x: qual_to_cluster.get(str(x), None))
        return df.dropna(subset=["cluster_id"]).assign(cluster_id=lambda d: d["cluster_id"].astype(int))

    df_v2 = assign_cluster(df_v)
    df_vd2 = assign_cluster(df_vd)
    both = pd.concat([df_v2, df_vd2], ignore_index=True)
    if both.empty:
        return
    both["n"] = pd.to_numeric(both["n"], errors="coerce").fillna(0).astype(int)
    both["accuracy"] = pd.to_numeric(both["accuracy"], errors="coerce").fillna(0.0)
    both["acc_n"] = both["accuracy"] * both["n"]
    agg = (
        both.groupby(["cluster_id", "model_id"], dropna=False)
        .agg(n=("n", "sum"), acc_n=("acc_n", "sum"))
        .reset_index()
    )
    agg["accuracy"] = np.where(agg["n"] > 0, agg["acc_n"] / agg["n"], np.nan)
    # Pivot models x clusters so models are on rows, clusters on columns (flipped orientation)
    agg["model_id"] = pd.Categorical(agg["model_id"], categories=models, ordered=True)
    p = agg.pivot_table(index="model_id", columns="cluster_id", values="accuracy", aggfunc="mean").sort_index()
    # Optionally drop by cluster IDs before renaming (now columns)
    if exclude_ids:
        try:
            p = p.drop(columns=[int(i) for i in exclude_ids if i is not None], errors="ignore")
        except Exception:
            pass
    # Replace column labels with cluster_names where available
    col_labels = [cluster_names.get(int(i), f"Cluster {int(i)+1:02d}") for i in p.columns]
    p.columns = col_labels
    # Optionally drop excluded cluster labels (now columns)
    if exclude_labels:
        excl = {e.strip().lower() for e in exclude_labels if e and str(e).strip()}
        keep_cols = [lbl for lbl in p.columns if str(lbl).strip().lower() not in excl]
        p = p.loc[:, keep_cols]
    # Build figure with both adequate width and height
    # Aim for near-square cells: height/width proportional to counts
    cell_size = 0.6
    fig_w = max(16, cell_size * p.shape[1] + 8)
    fig_h = max(16, cell_size * p.shape[0] + 8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    # Map full model ids to short names on the index before plotting
    p2 = p.copy()
    p2.index = [short_model_name(str(idx)) for idx in p2.index]
    sns.heatmap(p2, ax=ax, cmap="RdPu", vmin=0.0, vmax=1.0, cbar_kws={"label": "Validity"})
    fig.suptitle(
        f"Validity — Model × Differed Qualification Cluster ({dataset_label})",
        fontweight="bold",
        fontsize=36,
    )
    ax.set_xlabel("Qualification Cluster", fontweight="bold")
    ax.set_ylabel("Model", fontweight="bold")
    # Ensure model names on y-axis are horizontal for readability
    plt.setp(ax.get_yticklabels(), rotation=0, va="center")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)


def run_for_dataset(agg_dir: Path, out_dir: Path, dataset_label: str, exclude_cluster_labels: List[str] | None = None, reuse_cluster_cache_only: bool = False, exclude_cluster_ids: List[int] | None = None) -> None:
    print(f"[INFO] Generating figures for {dataset_label} from {agg_dir} → {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load aggregated JSONs
    df_overall = read_json_df(agg_dir / "validity_overall.json")
    df_job = read_json_df(agg_dir / "validity_by_job_title.json")
    df_pair_num = read_json_df(agg_dir / "validity_by_pair_type_num_differed.json")
    df_diffqual = read_json_df(agg_dir / "validity_by_differed_qualification.json")
    df_demo_overall = read_json_df(agg_dir / "validity_demographics_overall.json")
    df_fair_demo = read_json_df(agg_dir / "fairness_by_demographics.json")
    df_impl_demo = read_json_df(agg_dir / "implicit_fairness_by_demographics.json")
    df_diffqual_demo = read_json_df(agg_dir / "validity_demographics_by_differed_qualification.json")

    # Ensure necessary columns
    for d in (df_overall, df_job, df_pair_num, df_diffqual, df_demo_overall, df_fair_demo, df_impl_demo, df_diffqual_demo):
        if not d.empty:
            # Map model_id to shorthand immediately so full names never appear downstream
            d["model_id"] = d.get("model_id", "").astype(str).map(short_model_name)
            if "accuracy" not in d.columns:
                d["accuracy"] = np.nan
            if "n" not in d.columns:
                d["n"] = 0
            # Mirror shorthand into model_short for plotting
            d["model_short"] = d["model_id"]

    # Model order based on overall (fallback: alphabetical)
    if not df_overall.empty:
        models = df_overall.sort_values("model_id")["model_id"].unique().tolist()
    else:
        models = sorted(pd.concat([df_job.get("model_id", pd.Series(dtype=str)), df_pair_num.get("model_id", pd.Series(dtype=str))], ignore_index=True).dropna().unique().tolist())
    models_short = [short_model_name(m) for m in models]

    set_style()

    # 1) Validity vs k per model (3x3 grid)
    try:
        # Recompute validity by num_differed from baseline CSVs restricted to same-demographic pairs
        base_dir = agg_dir.parents[2] / "evaluations" / "baseline"
        df_pair_num_same = compute_validity_by_num_from_baseline_same_demo(base_dir, dataset_label)
        if not df_pair_num_same.empty:
            plot_validity_vs_k_grid(df_pair_num_same, models, out_dir / "validity_vs_k_grid.png", dataset_label)
        else:
            # Fallback to aggregated if same-demo baseline not available
            plot_validity_vs_k_grid(df_pair_num, models, out_dir / "validity_vs_k_grid.png", dataset_label)
    except Exception:
        print("[ERROR] plotting validity vs k failed")
        print(traceback.format_exc())
    # 1b) Combined Claude vs Gemini for each model, only when both dirs exist
    try:
        # Infer sibling agg dir (claude <-> gemini)
        sibling = None
        if str(agg_dir).endswith("/claude"):
            sibling = agg_dir.parent / "gemini"
            label_a, label_b = "Claude", "Gemini"
        elif str(agg_dir).endswith("/gemini"):
            sibling = agg_dir.parent / "claude"
            label_a, label_b = "Gemini", "Claude"
        if sibling and (sibling / "validity_by_pair_type_num_differed.json").exists():
            # Build same-demo baselines for the sibling too
            base_dir = agg_dir.parents[2] / "evaluations" / "baseline"
            other_same = compute_validity_by_num_from_baseline_same_demo(base_dir, label_b)
            if other_same.empty:
                other_same = read_json_df(sibling / "validity_by_pair_type_num_differed.json")
            # Use same models list for alignment
            base_dir_self = agg_dir.parents[2] / "evaluations" / "baseline"
            self_same = compute_validity_by_num_from_baseline_same_demo(base_dir_self, label_a)
            if self_same.empty:
                self_same = df_pair_num
            plot_validity_vs_k_grid_combined(self_same, other_same, models, out_dir / "validity_vs_k_grid_combined.png", label_a, label_b)
    except Exception:
        print("[ERROR] plotting combined validity vs k failed")
        print(traceback.format_exc())

    # 1.5) Generate LaTeX table for validity vs k grid (combined Claude and Gemini)
    try:
        print("[INFO] generating LaTeX table for validity vs k grid (combined)")
        
        # Load both Claude and Gemini data
        claude_df = read_json_df(Path("/home/zs7353/resume_validity/analysis/aggregates_rel6_json/claude/validity_by_pair_type_num_differed.json"))
        gemini_df = read_json_df(Path("/home/zs7353/resume_validity/analysis/aggregates_rel6_json/gemini/validity_by_pair_type_num_differed.json"))
        
        if not claude_df.empty and not gemini_df.empty:
            # Add dataset labels
            claude_df["dataset"] = "Claude"
            gemini_df["dataset"] = "Gemini"
            
            # Combine data
            combined_df = pd.concat([claude_df, gemini_df], ignore_index=True)
            
            # Create pivot table with dataset and model as index
            pivot_table = combined_df.pivot_table(
                index=["dataset", "model_id"], 
                columns="num_differed", 
                values="accuracy", 
                aggfunc="mean"
            ).round(3)
            
            # Generate LaTeX table
            latex_table = "\\begin{table}[h]\n"
            latex_table += "\\centering\n"
            latex_table += "\\caption{Validity by Number of Differed Qualifications: Claude vs Gemini}\n"
            latex_table += "\\label{tab:validity_vs_k_combined}\n"
            latex_table += "\\begin{tabular}{ll" + "c" * len(pivot_table.columns) + "}\n"
            latex_table += "\\toprule\n"
            
            # Header row
            latex_table += "Dataset & Model"
            for k in pivot_table.columns:
                latex_table += f" & k={k}"
            latex_table += " \\\\\n"
            latex_table += "\\midrule\n"
            
            # Data rows
            current_dataset = None
            for (dataset, model) in pivot_table.index:
                if dataset != current_dataset:
                    current_dataset = dataset
                    latex_table += f"\\multirow{{{len([m for d, m in pivot_table.index if d == dataset])}}}{{*}}{{{dataset}}}"
                else:
                    latex_table += " &"
                
                model_escaped = model.replace('_', '\\_')
                latex_table += f" & {model_escaped}"
                for k in pivot_table.columns:
                    value = pivot_table.loc[(dataset, model), k]
                    if pd.isna(value):
                        latex_table += " & --"
                    else:
                        latex_table += f" & {value:.3f}"
                latex_table += " \\\\\n"
            
            latex_table += "\\bottomrule\n"
            latex_table += "\\end{tabular}\n"
            latex_table += "\\end{table}\n"
            
            # Print LaTeX table command for copy-paste
            print(f"[INFO] LaTeX table command for copy-paste:")
            print("=" * 80)
            print(latex_table)
            print("=" * 80)
            
    except Exception:
        print("[ERROR] generating LaTeX table for validity vs k grid failed")
        print(traceback.format_exc())

    # 2) Validity by pair type (one bar plot with triplets per model)
    try:
        plot_pair_type_bars(df_pair_num, models, out_dir / "validity_by_pair_type_bars.png", dataset_label)
    except Exception:
        print("[ERROR] plotting pair-type bars failed")
        print(traceback.format_exc())

    # 3) Fairness bar plots: over- and under-assessment rates per demographic combo × model
    try:
        if not df_fair_demo.empty:
            # Load raw baseline rows to compute directional misassessments (group-level)
            base_dir = agg_dir.parents[2] / "evaluations" / "baseline"
            includes, excludes = eval_file_filters_for_dataset(dataset_label)
            fair_rows = load_baseline_df(base_dir, includes=includes, excludes=excludes)
            fair_rows = fair_rows[fair_rows["experiment_type"] == "fairness"].copy()
            # Explicitly restrict to mixed pairs among canonical groups (Option A)
            allowed = {"B_W", "B_M", "W_W", "W_M"}
            fair_rows = fair_rows[(fair_rows["demographic_base"].isin(allowed)) & (fair_rows["demographic_variant"].isin(allowed)) & (fair_rows["demographic_base"] != fair_rows["demographic_variant"])].copy()
            # Build decisions with abstain override
            dec_raw = fair_rows["decision"].astype(str).str.lower()
            dec = dec_raw.where(~fair_rows["abstained"].astype(bool), other="abstain")
            # Over-/Under-assessment using ground-truth 'better' (first/second/equal)
            gt = fair_rows["better"].astype(str).str.lower()
            # Over: group has worse resume but judged better or abstained; for equal-case, only counted when judged better
            fair_rows["over_base"] = ((gt == "second") & dec.isin(["first", "abstain"])) | ((gt == "equal") & dec.eq("first"))
            fair_rows["over_variant"] = ((gt == "first") & dec.isin(["second", "abstain"])) | ((gt == "equal") & dec.eq("second"))
            # Under: group has better resume but judged worse or abstained; for equal-case, only counted when judged worse
            fair_rows["under_base"] = ((gt == "first") & dec.isin(["second", "abstain"])) | ((gt == "equal") & dec.eq("second"))
            fair_rows["under_variant"] = ((gt == "second") & dec.isin(["first", "abstain"])) | ((gt == "equal") & dec.eq("first"))
            # Aggregate per demographic group (not pairs): we count over/under for the role a group plays (base or variant)
            def agg_group(flag_col: str, demo_col: str):
                num = fair_rows[fair_rows[flag_col]].groupby(["model_id", demo_col]).size().rename("num").reset_index()
                den = fair_rows.groupby(["model_id", demo_col]).size().rename("den").reset_index()
                m = pd.merge(den, num, on=["model_id", demo_col], how="left").fillna({"num": 0})
                m["rate"] = np.where(m["den"] > 0, m["num"] / m["den"], np.nan)
                m["group"] = demo_col
                m = m.rename(columns={demo_col: "demographic"})
                m["model_short"] = m["model_id"].map(short_model_name)
                return m
            over_g = pd.concat([agg_group("over_base", "demographic_base"), agg_group("over_variant", "demographic_variant")], ignore_index=True)
            under_g = pd.concat([agg_group("under_base", "demographic_base"), agg_group("under_variant", "demographic_variant")], ignore_index=True)
            # Plot per demographic group (x=demographic, hue=model)
            for name, ddf, fname in [("Over-assessment", over_g, "fairness_over_assessment_bars.png"), ("Under-assessment", under_g, "fairness_under_assessment_bars.png")]:
                if ddf.empty:
                    raise RuntimeError(f"Empty data for {name} (fairness)")
                # Wider figure and increased spacing between model groups
                fig, ax = plt.subplots(figsize=(16, 7))
                # compute SE for rates per model×demographic
                ddf2 = ddf.copy()
                ddf2["se"] = np.sqrt(ddf2["rate"] * (1 - ddf2["rate"]) / ddf2["den"].replace(0, np.nan))
                sns.barplot(data=ddf2, x="model_short", y="rate", hue="demographic", ax=ax, errorbar=None)
                # increase distance between neighboring sets
                for c in ax.containers:
                    for bar in c:
                        bar.set_width(bar.get_width() * 0.8)
                ax.set_ylim(0.0, 1.0)
                ax.set_xlabel("Model")
                ax.set_ylabel("Rate")
                ax.set_title(f"{name} — by Demographic Group ({dataset_label})")
                # add error bars manually
                xticks = [t.get_text() for t in ax.get_xticklabels()]
                hue_vals = sorted(ddf2["demographic"].unique())
                bar_idx = 0
                for ms in xticks:
                    for hv in hue_vals:
                        row = ddf2[(ddf2["model_short"] == ms) & (ddf2["demographic"] == hv)].head(1)
                        if not row.empty:
                            se = float(row["se"].iloc[0]) if not pd.isna(row["se"].iloc[0]) else np.nan
                            if not np.isnan(se) and bar_idx < len(ax.patches):
                                bar = ax.patches[bar_idx]
                                x = bar.get_x() + bar.get_width() / 2
                                y = bar.get_height()
                                ax.errorbar(x, y, yerr=se, ecolor="#333333", capsize=3, fmt='none', lw=1)
                        bar_idx += 1
                leg = ax.legend(title="Demographic", loc="upper right", framealpha=0.4)
                if leg and leg.get_frame():
                    leg.get_frame().set_alpha(0.4)
                plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
                fig.tight_layout()
                fig.savefig(out_dir / fname)
                plt.close(fig)
            # finished fairness group-level plots above
    except Exception:
        print("[ERROR] plotting fairness bar plots failed")
        print(traceback.format_exc())

    # 4) Implicit fairness bar plots: over- and under-assessment rates
    try:
        if not df_impl_demo.empty:
            base_dir = agg_dir.parents[2] / "evaluations" / "baseline"
            includes, excludes = eval_file_filters_for_dataset(dataset_label)
            impl_rows = load_baseline_df(base_dir, includes=includes, excludes=excludes)
            impl_rows = impl_rows[impl_rows["experiment_type"] == "implicit_demographics_fairness"].copy()
            allowed = {"B_W", "B_M", "W_W", "W_M"}
            impl_rows = impl_rows[(impl_rows["demographic_base"].isin(allowed)) & (impl_rows["demographic_variant"].isin(allowed)) & (impl_rows["demographic_base"] != impl_rows["demographic_variant"])].copy()
            impl_rows["demo_combo"] = impl_rows["demographic_base"].astype(str) + " / " + impl_rows["demographic_variant"].astype(str)
            # Group-level over/under for implicit fairness using same logic as fairness
            dec_raw = impl_rows["decision"].astype(str).str.lower()
            dec = dec_raw.where(~impl_rows["abstained"].astype(bool), other="abstain")
            gt = impl_rows["better"].astype(str).str.lower()
            impl_rows["over_base"] = ((gt == "second") & dec.isin(["first", "abstain"])) | ((gt == "equal") & dec.eq("first"))
            impl_rows["over_variant"] = ((gt == "first") & dec.isin(["second", "abstain"])) | ((gt == "equal") & dec.eq("second"))
            impl_rows["under_base"] = ((gt == "first") & dec.isin(["second", "abstain"])) | ((gt == "equal") & dec.eq("second"))
            impl_rows["under_variant"] = ((gt == "second") & dec.isin(["first", "abstain"])) | ((gt == "equal") & dec.eq("first"))
            def agg_group2(flag_col: str, demo_col: str):
                num = impl_rows[impl_rows[flag_col]].groupby(["model_id", demo_col]).size().rename("num").reset_index()
                den = impl_rows.groupby(["model_id", demo_col]).size().rename("den").reset_index()
                m = pd.merge(den, num, on=["model_id", demo_col], how="left").fillna({"num": 0})
                m["rate"] = np.where(m["den"] > 0, m["num"] / m["den"], np.nan)
                m["group"] = demo_col
                m = m.rename(columns={demo_col: "demographic"})
                m["model_short"] = m["model_id"].map(short_model_name)
                return m
            over_g = pd.concat([agg_group2("over_base", "demographic_base"), agg_group2("over_variant", "demographic_variant")], ignore_index=True)
            under_g = pd.concat([agg_group2("under_base", "demographic_base"), agg_group2("under_variant", "demographic_variant")], ignore_index=True)
            for name, ddf, fname in [("Implicit Over-assessment", over_g, "implicit_fairness_over_assessment_bars.png"), ("Implicit Under-assessment", under_g, "implicit_fairness_under_assessment_bars.png")]:
                if ddf.empty:
                    raise RuntimeError(f"Empty data for {name} (implicit fairness)")
                fig, ax = plt.subplots(figsize=(16, 7))
                ddf2 = ddf.copy()
                ddf2["se"] = np.sqrt(ddf2["rate"] * (1 - ddf2["rate"]) / ddf2["den"].replace(0, np.nan))
                sns.barplot(data=ddf2, x="model_short", y="rate", hue="demographic", ax=ax, errorbar=None)
                for c in ax.containers:
                    for bar in c:
                        bar.set_width(bar.get_width() * 0.8)
                ax.set_ylim(0.0, 1.0)
                ax.set_xlabel("Model")
                ax.set_ylabel("Rate")
                ax.set_title(f"{name} — by Demographic Group ({dataset_label})")
                # add error bars manually
                xticks = [t.get_text() for t in ax.get_xticklabels()]
                hue_vals = sorted(ddf2["demographic"].unique())
                bar_idx = 0
                for ms in xticks:
                    for hv in hue_vals:
                        row = ddf2[(ddf2["model_short"] == ms) & (ddf2["demographic"] == hv)].head(1)
                        if not row.empty:
                            se = float(row["se"].iloc[0]) if not pd.isna(row["se"].iloc[0]) else np.nan
                            if not np.isnan(se) and bar_idx < len(ax.patches):
                                bar = ax.patches[bar_idx]
                                x = bar.get_x() + bar.get_width() / 2
                                y = bar.get_height()
                                ax.errorbar(x, y, yerr=se, ecolor="#333333", capsize=3, fmt='none', lw=1)
                        bar_idx += 1
                leg = ax.legend(title="Demographic", loc="upper right", framealpha=0.4)
                if leg and leg.get_frame():
                    leg.get_frame().set_alpha(0.4)
                plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
                fig.tight_layout()
                fig.savefig(out_dir / fname)
                plt.close(fig)
    except Exception:
        print("[ERROR] plotting implicit fairness bar plots failed")
        print(traceback.format_exc())

    # 5) Job × model heatmap — models on rows, jobs on columns; increase height to fit job labels
    try:
        if not df_job.empty:
            d = df_job[["model_id", "job_title", "accuracy"]].copy()
            d["model_short"] = d["model_id"].astype(str).map(short_model_name)
            # For job heatmap, disable square cells and compute a balanced width to avoid side whitespace
            num_models = max(1, d["model_short"].nunique())
            num_jobs = max(1, d["job_title"].nunique())
            # Target aspect: wider when many jobs, taller when many models
            # Increase dimensions significantly to prevent truncation and reduce whitespace
            cell_w, cell_h = 0.8, 1.2
            fig_w = max(20, cell_w * num_jobs + 8)
            fig_h = max(24, cell_h * num_models + 12)
            # Plot job heatmap exactly like qualification heatmap
            p = d.pivot_table(index="model_short", columns="job_title", values="accuracy", aggfunc="mean").sort_index()
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            sns.heatmap(p, ax=ax, cmap="RdPu", vmin=0.0, vmax=1.0, cbar_kws={"label": "Validity"})
            fig.suptitle(
                f"Validity — Model × Job ({dataset_label})",
                fontweight="bold",
                fontsize=48,
            )
            ax.set_xlabel("Job", fontweight="bold", fontsize=28)
            ax.set_ylabel("Model", fontweight="bold", fontsize=28)
            # Ensure model names on y-axis are horizontal for readability
            plt.setp(ax.get_yticklabels(), rotation=0, va="center")
            plt.setp(ax.get_xticklabels(), rotation=90, ha="right")
            # Make tick labels larger
            ax.tick_params(axis="both", labelsize=20)
            # Colorbar tick and label sizes
            try:
                cbar = ax.collections[0].colorbar
                if cbar is not None:
                    cbar.ax.tick_params(labelsize=18)
                    cbar.ax.yaxis.label.set_size(28)
            except Exception:
                pass
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            fig.savefig(out_dir / "validity_by_job_by_model.png")
            plt.close(fig)
    except Exception:
        print("[ERROR] plotting job × model heatmap failed")
        print(traceback.format_exc())

    # 5.5) Relative Error Rate by Demographic Groups
    try:
        print("[INFO] plotting relative error rate by demographic groups")
        
        # Load raw CSV files from baseline evaluations
        base_dir = agg_dir.parents[2] / "evaluations" / "baseline"
        # For Gemini dataset, we need to load both gemini_rel6 and claude_rel6 files
        # because most models were evaluated on claude_rel6 files
        if dataset_label.lower() == "gemini":
            includes = ["*gemini_rel6_shard*.csv", "*claude_rel6_shard*.csv"]
            excludes = []
        else:
            includes = [f"*{dataset_label.lower()}_rel6_shard*.csv"]
            excludes = [f"*claude_rel6_shard*.csv" if dataset_label.lower() != "claude" else ""]
        
        # Load all CSV files with model identification
        all_data = []
        for include_pattern in includes:
            for fp in base_dir.glob(include_pattern):
                if any(exclude in str(fp) for exclude in excludes if exclude):
                    continue
                try:
                    df = pd.read_csv(fp)
                    # Filter for validity and validity_demographics experiments
                    df = df[df["experiment_type"].isin(["validity", "validity_demographics"])]
                    if not df.empty:
                        # Extract model name from filename
                        model_name = fp.name.split("_")[0]
                        df["model_id"] = model_name
                        all_data.append(df)
                except Exception as e:
                    print(f"[WARNING] Failed to load {fp}: {e}")
                    continue
        
        if not all_data:
            print("[WARNING] No raw CSV data found for relative error calculation")
            return
            
        df_raw = pd.concat(all_data, ignore_index=True)
        
        # Define demographic groups
        demographic_groups = ["B_W", "B_M", "W_W", "W_M"]
        
        # Calculate relative error rates
        # Relative error rate = conditional error rate: among cases where model didn't pick better candidate,
        # what fraction picked the worse candidate?
        # Formula: sum(choice = worse) / (sum(abstentions) + sum(choice = worse))
        
        # Same demographic group cases
        same_demo_data = []
        for model in df_raw["model_id"].unique():
            model_data = df_raw[df_raw["model_id"] == model]
            same_demo_rows = model_data[
                (model_data["demographic_base"] == model_data["demographic_variant"]) &
                (model_data["demographic_base"].isin(demographic_groups))
                # Don't filter out abstentions - we need them for the calculation
            ]
            if len(same_demo_rows) > 0:
                # Filter for cases where resumes are unequal (better != 'equal')
                unequal_rows = same_demo_rows[same_demo_rows["better"] != "equal"]
                
                if len(unequal_rows) > 0:
                    # Count abstentions and wrong choices
                    abstentions = (unequal_rows["abstained"] == True).sum()
                    wrong_choices = (unequal_rows["decision"] != unequal_rows["better"]).sum()
                    
                    # Relative error rate = wrong choices / (abstentions + wrong choices)
                    # This is conditional on model not picking the better candidate
                    denominator = abstentions + wrong_choices
                    relative_error = wrong_choices / denominator if denominator > 0 else 0
                    se = np.sqrt(relative_error * (1 - relative_error) / denominator) if denominator > 0 else np.nan
                    
                    same_demo_data.append({
                        "model_id": model,
                        "demographic_type": "Same Demographic",
                        "wrong_choices": int(wrong_choices),
                        "abstentions": int(abstentions),
                        "total_non_correct": int(denominator),
                        "relative_error": float(relative_error),
                        "se": float(se) if not np.isnan(se) else np.nan,
                    })
        
        # Cross-demographic cases: when one demographic group has better resume
        cross_demo_data = []
        for model in df_raw["model_id"].unique():
            model_data = df_raw[df_raw["model_id"] == model]
            for demo_group in demographic_groups:
                # Cases where this demographic group has the better resume
                # Look for pairs where this group is base and variant is different
                cross_rows = model_data[
                    (model_data["demographic_base"] == demo_group) &
                    (model_data["demographic_variant"] != demo_group) &
                    (model_data["demographic_variant"].isin(demographic_groups))
                    # Don't filter out abstentions - we need them for the calculation
                ]
                if len(cross_rows) > 0:
                    # Filter for cases where resumes are unequal (better != 'equal')
                    unequal_rows = cross_rows[cross_rows["better"] != "equal"]
                    
                    if len(unequal_rows) > 0:
                        # Count abstentions and wrong choices
                        abstentions = (unequal_rows["abstained"] == True).sum()
                        wrong_choices = (unequal_rows["decision"] != unequal_rows["better"]).sum()
                        
                        # Relative error rate = wrong choices / (abstentions + wrong choices)
                        # This is conditional on model not picking the better candidate
                        denominator = abstentions + wrong_choices
                        relative_error = wrong_choices / denominator if denominator > 0 else 0
                        se = np.sqrt(relative_error * (1 - relative_error) / denominator) if denominator > 0 else np.nan
                        
                        cross_demo_data.append({
                            "model_id": model,
                            "demographic_type": f"Better Resume: {demo_group}",
                            "wrong_choices": int(wrong_choices),
                            "abstentions": int(abstentions),
                            "total_non_correct": int(denominator),
                            "relative_error": float(relative_error),
                            "se": float(se) if not np.isnan(se) else np.nan,
                        })
        
        # Combine data
        all_data = same_demo_data + cross_demo_data
        df_relative_error = pd.DataFrame(all_data)
        
        if len(df_relative_error) > 0:
            # Add short model names
            df_relative_error["model_short"] = df_relative_error["model_id"].map(short_model_name)
            # Enforce consistent model order matching other bar plots
            try:
                df_relative_error["model_short"] = pd.Categorical(
                    df_relative_error["model_short"], categories=models_short, ordered=True
                )
            except Exception:
                pass
            
            # Create plot using seaborn barplot (consistent with other plots)
            fig, ax = plt.subplots(figsize=(20, 12))
            
            # Use seaborn barplot for consistent styling
            hue_order = [
                "Same Demographic",
                "Better Resume: B_W",
                "Better Resume: B_M",
                "Better Resume: W_W",
                "Better Resume: W_M",
            ]
            sns.barplot(
                data=df_relative_error, 
                x="model_short", 
                y="relative_error", 
                hue="demographic_type",
                hue_order=hue_order,
                ax=ax, 
                errorbar=None
            )
            # shrink bar widths and add x margin to reduce label clash (robust)
            for rect in ax.patches:
                if hasattr(rect, "get_width") and hasattr(rect, "set_width"):
                    try:
                        rect.set_width(rect.get_width() * 0.8)
                    except Exception:
                        pass
            ax.margins(x=0.06)
            
            ax.set_xlabel("Model", fontweight="bold", fontsize=20)
            ax.set_ylabel("Relative Error Rate", fontweight="bold", fontsize=20)
            ax.set_title(f"Relative Error Rate by Demographic Groups ({dataset_label})", fontweight="bold", fontsize=24)
            
            # Set x-axis labels to horizontal (consistent with other plots)
            plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
            
            # Add manual error bars matching seaborn bar order
            xticks = [t.get_text() for t in ax.get_xticklabels()]
            bar_idx = 0
            for ms in xticks:
                for hv in hue_order:
                    row = df_relative_error[(df_relative_error["model_short"].astype(str) == ms) & (df_relative_error["demographic_type"] == hv)].head(1)
                    if not row.empty and bar_idx < len(ax.patches):
                        se_val = row["se"].iloc[0]
                        if pd.notna(se_val):
                            bar = ax.patches[bar_idx]
                            x = bar.get_x() + bar.get_width() / 2
                            y = bar.get_height()
                            ax.errorbar(x, y, yerr=se_val, ecolor="#333333", capsize=3, fmt='none', lw=1)
                    bar_idx += 1

            # Legend styling (consistent with other plots)
            leg = ax.legend(title="Demographic Type", loc="upper right", framealpha=0.4)
            if leg and leg.get_frame():
                leg.get_frame().set_alpha(0.4)
            
            ax.grid(True, alpha=0.3)
            
            # Set y-axis to show full range
            ax.set_ylim(0, 1.0)
            
            fig.tight_layout()
            fig.savefig(out_dir / "relative_error_rate_by_demographic.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            
    except Exception:
        print("[ERROR] plotting relative error rate by demographic groups failed")
        print(traceback.format_exc())

    # 6) Validity vs Validity_Demographics (pairs of bars per model)
    try:
        if not (df_overall.empty and df_demo_overall.empty):
            plot_valid_vs_valid_demo(df_overall, df_demo_overall, models, out_dir / "validity_vs_validity_demographics_bars.png", dataset_label)
            # Conditional non-abstain validity bars (preferred vs underqualified, excluding reworded)
            plot_pair_type_bars_noabstain(models, out_dir / "validity_nonabstain_preferred_vs_underqualified_bars.png", dataset_label, agg_dir)
    except Exception:
        print("[ERROR] plotting validity vs validity_demographics failed")
        print(traceback.format_exc())

    # 7) Semantic clustering of differed qualifications → 20 clusters heatmap
    try:
        cluster_diff_quals_and_plot(
            df_diffqual[["model_id", "differed_qualification", "n", "accuracy"]] if not df_diffqual.empty else pd.DataFrame(columns=["model_id", "differed_qualification", "n", "accuracy"]),
            df_diffqual_demo[["model_id", "differed_qualification", "n", "accuracy"]] if not df_diffqual_demo.empty else pd.DataFrame(columns=["model_id", "differed_qualification", "n", "accuracy"]),
            models,
            out_dir / "validity_by_qualification_cluster_by_model.png",
            dataset_label,
            n_clusters=20,
            exclude_labels=exclude_cluster_labels,
            reuse_cache_only=reuse_cluster_cache_only,
            exclude_ids=exclude_cluster_ids,
        )
    except Exception:
        print("[ERROR] plotting qualification-cluster heatmap failed")
        print(traceback.format_exc())

    # 8) Additional bar: non-abstention rates in preferred vs underqualified (validity_experiment only)
    try:
        base_dir = agg_dir.parents[2] / "evaluations" / "baseline"
        includes, excludes = eval_file_filters_for_dataset(dataset_label)
        rows = load_baseline_df(base_dir, includes=includes, excludes=excludes)
        rows = rows[rows["experiment_type"].isin(["validity", "validity_demographics"])].copy()
        rows["non_abstain"] = ~rows["abstained"]
        sub = rows[rows["pair_type"].isin(["preferred", "underqualified"])].copy()
        g = sub.groupby(["model_id", "pair_type"]).agg(n=("non_abstain", "size"), k=("non_abstain", "sum")).reset_index()
        g["rate"] = np.where(g["n"] > 0, g["k"] / g["n"], np.nan)
        g["model_short"] = g["model_id"].map(short_model_name)
        fig, ax = plt.subplots(figsize=(16, 7))
        g = g.copy()
        g["se"] = np.sqrt(g["rate"] * (1 - g["rate"]) / g["n"].replace(0, np.nan))
        sns.barplot(data=g, x="model_short", y="rate", hue="pair_type", ax=ax, errorbar=None)
        for c in ax.containers:
            for bar in c:
                bar.set_width(bar.get_width() * 0.8)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Model")
        ax.set_ylabel("Non-abstention rate")
        ax.set_title(f"Non-abstention in Preferred vs Underqualified ({dataset_label})")
        add_random_baseline(ax, y=2/3, label="random baseline (2/3)")
        # add error bars manually
        xticks = [t.get_text() for t in ax.get_xticklabels()]
        hue_vals = sorted(g["pair_type"].unique())
        bar_idx = 0
        for ms in xticks:
            for hv in hue_vals:
                row = g[(g["model_short"] == ms) & (g["pair_type"] == hv)].head(1)
                if not row.empty:
                    se = float(row["se"].iloc[0]) if not pd.isna(row["se"].iloc[0]) else np.nan
                    if not np.isnan(se) and bar_idx < len(ax.patches):
                        bar = ax.patches[bar_idx]
                        x = bar.get_x() + bar.get_width() / 2
                        y = bar.get_height()
                        ax.errorbar(x, y, yerr=se, ecolor="#333333", capsize=3, fmt='none', lw=1)
                bar_idx += 1
        leg = ax.legend(title="Pair Type", loc="upper right", framealpha=0.4)
        if leg and leg.get_frame():
            leg.get_frame().set_alpha(0.4)
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        fig.tight_layout()
        fig.savefig(out_dir / "non_abstention_preferred_vs_underqualified_bars.png")
        plt.close(fig)
    except Exception:
        print("[ERROR] plotting non-abstention preferred vs underqualified failed")
        print(traceback.format_exc())

    # 9) Additional bar: abstention in implicit vs explicit fairness
    try:
        base_dir = agg_dir.parents[2] / "evaluations" / "baseline"
        includes, excludes = eval_file_filters_for_dataset(dataset_label)
        rows = load_baseline_df(base_dir, includes=includes, excludes=excludes)
        sub = rows[rows["experiment_type"].isin(["fairness", "implicit_demographics_fairness"])].copy()
        sub["abstained_bool"] = sub["abstained"].astype(bool)
        g = sub.groupby(["model_id", "experiment_type"]).agg(n=("abstained_bool", "size"), k=("abstained_bool", "sum")).reset_index()
        g["rate"] = np.where(g["n"] > 0, g["k"] / g["n"], np.nan)
        g["model_short"] = g["model_id"].map(short_model_name)
        g["exp_short"] = g["experiment_type"].map({"fairness": "explicit_fairness", "implicit_demographics_fairness": "implicit_fairness"})
        fig, ax = plt.subplots(figsize=(16, 7))
        g = g.copy()
        g["se"] = np.sqrt(g["rate"] * (1 - g["rate"]) / g["n"].replace(0, np.nan))
        sns.barplot(data=g, x="model_short", y="rate", hue="exp_short", ax=ax, errorbar=None)
        for c in ax.containers:
            for bar in c:
                bar.set_width(bar.get_width() * 0.8)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Model")
        ax.set_ylabel("Abstention rate")
        ax.set_title(f"Abstention: Implicit vs Explicit Fairness ({dataset_label})")
        add_random_baseline(ax, y=1/3, label="random baseline (1/3)")
        # add error bars manually
        xticks = [t.get_text() for t in ax.get_xticklabels()]
        hue_vals = sorted(g["exp_short"].unique())
        bar_idx = 0
        for ms in xticks:
            for hv in hue_vals:
                row = g[(g["model_short"] == ms) & (g["exp_short"] == hv)].head(1)
                if not row.empty:
                    se = float(row["se"].iloc[0]) if not pd.isna(row["se"].iloc[0]) else np.nan
                    if not np.isnan(se) and bar_idx < len(ax.patches):
                        bar = ax.patches[bar_idx]
                        x = bar.get_x() + bar.get_width() / 2
                        y = bar.get_height()
                        ax.errorbar(x, y, yerr=se, ecolor="#333333", capsize=3, fmt='none', lw=1)
                bar_idx += 1
        leg = ax.legend(title="Experiment", loc="upper right", framealpha=0.4)
        if leg and leg.get_frame():
            leg.get_frame().set_alpha(0.4)
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        fig.tight_layout()
        fig.savefig(out_dir / "abstention_implicit_vs_explicit_fairness_bars.png")
        plt.close(fig)
    except Exception:
        print("[ERROR] plotting abstention implicit vs explicit failed")
        print(traceback.format_exc())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Plot aggregated results for Claude and Gemini datasets")
    p.add_argument("--agg_dir", type=str, required=True, help="Directory with aggregated JSON files (e.g., analysis/aggregates_rel6_json/claude)")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for figures")
    p.add_argument("--label", type=str, required=True, help="Dataset label to show on plots (e.g., Claude rel6)")
    p.add_argument("--exclude_cluster_labels", type=str, nargs="*", default=[], help="Cluster labels to exclude from the qualification heatmap (exact match, case-insensitive)")
    p.add_argument("--reuse_cluster_cache_only", action="store_true", help="Reuse existing cached Gemini cluster labels without making API calls; hard-fail if any are missing")
    p.add_argument("--exclude_cluster_ids", type=int, nargs="*", default=[], help="Cluster IDs to exclude from the qualification heatmap (by numeric cluster index)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run_for_dataset(Path(args.agg_dir), Path(args.out_dir), args.label, exclude_cluster_labels=args.exclude_cluster_labels, reuse_cluster_cache_only=args.reuse_cluster_cache_only, exclude_cluster_ids=args.exclude_cluster_ids)
    except Exception:
        # As per user preference, always print full exception and trace
        print("[FATAL] Unhandled exception in plotting script")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()


