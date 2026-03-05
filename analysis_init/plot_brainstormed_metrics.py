#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Plot brainstormed metrics (criterion validity, unjustified abstention, discriminant validity, selection rate)")
    p.add_argument("--metrics_dir", type=str, required=True, help="Directory with metrics JSONs (output of compute_metrics_from_results.py)")
    p.add_argument("--out_dir", type=str, required=True, help="Directory to write figures")
    return p.parse_args()


def read_json(path: Path) -> pd.DataFrame:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except Exception:
        print(f"[ERROR] Failed reading {path}")
        print(traceback.format_exc())
        return pd.DataFrame()


def short_model_name(m: str) -> str:
    mapping = {
        "anthropic/claude-sonnet-4": "claude",
        "deepseek/deepseek-chat-v3.1": "deepseek",
        "google/gemini-2.0-flash-001": "gemini-2.0",
        "google/gemini-2.5-pro": "gemini-2.5",
        "google/gemma-3-12b-it": "gemma",
        "openai/gpt-4o-mini": "gpt-4o-mini",
        "openai/gpt-5": "gpt-5",
        "meta-llama/llama-3.1-8b-instruct": "llama-8b",
        "meta-llama/llama-3.3-70b-instruct": "llama-70b",
        # filename-derived fallbacks
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
    return mapping.get(m, m)


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.4)
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 120,
        "axes.titlesize": 24,
        "axes.titleweight": "bold",
        "axes.labelsize": 18,
        "axes.labelweight": "bold",
        "legend.fontsize": 12,
        "legend.title_fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "font.weight": "bold",
    })


def add_baseline(ax, y: float, label: str) -> None:
    ax.axhline(y, ls="--", lw=1.2, color="#666666")
    ax.text(0.99, y + 0.01, label, color="#666666", fontsize=10, ha="right", va="bottom", transform=ax.get_yaxis_transform())


def weighted_mean(df: pd.DataFrame, value_col: str, weight_col: str, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    d[weight_col] = pd.to_numeric(d[weight_col], errors="coerce").fillna(0).astype(float)
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce").astype(float)
    d["num"] = d[value_col] * d[weight_col]
    g = d.groupby(group_cols, dropna=False).agg(num=("num", "sum"), den=(weight_col, "sum")).reset_index()
    g[value_col] = np.where(g["den"] > 0, g["num"] / g["den"], np.nan)
    return g.drop(columns=["num", "den"])


def bars_by_k(df: pd.DataFrame, value_col: str, weight_col: str, out_path: Path, constructed_by: str, title: str, baseline: float | None = None, baseline_label: str | None = None) -> None:
    if df.empty:
        return
    d = df.copy()
    d = d[d["constructed_by"].astype(str).str.lower() == constructed_by]
    if d.empty:
        return
    # Aggregate across frameworks at (model_id, k), and compute total weight per group for SE
    den_all = None
    if "k" in d.columns:
        den_all = d.groupby(["model_id", "k"], dropna=False)[weight_col].sum().rename("den").reset_index()
        d = weighted_mean(d, value_col=value_col, weight_col=weight_col, group_cols=["model_id", "k"]).merge(den_all, on=["model_id", "k"], how="left")
    d["model_short"] = d["model_id"].astype(str).map(short_model_name)
    ks = sorted([int(x) for x in d.get("k", pd.Series([0])).dropna().unique().tolist()])
    if not ks:
        ks = [0]
    n = len(ks)
    rows = n
    cols = 1
    fig_w = 18
    fig_h = max(6 * rows, 8)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)
    axes_flat = axes.flatten()
    for idx, k in enumerate(ks):
        ax = axes_flat[idx]
        dk = d[d.get("k", 0) == k].copy()
        if dk.empty:
            ax.axis('off')
            continue
        # Compute SE if denominator available
        if "den" in dk.columns:
            dk["se"] = np.where(dk["den"] > 0, np.sqrt(dk[value_col] * (1 - dk[value_col]) / dk["den"]), np.nan)
        dk = dk.sort_values("model_short")
        sns.barplot(data=dk, x="model_short", y=value_col, ax=ax, errorbar=None)
        # Overlay error bars
        if "se" in dk.columns:
            xticks = [t.get_text() for t in ax.get_xticklabels()]
            bar_idx = 0
            for ms in xticks:
                row = dk[dk["model_short"] == ms].head(1)
                if not row.empty and pd.notna(row["se"].iloc[0]):
                    if bar_idx < len(ax.patches):
                        bar = ax.patches[bar_idx]
                        x = bar.get_x() + bar.get_width() / 2
                        y = bar.get_height()
                        ax.errorbar(x, y, yerr=float(row["se"].iloc[0]), ecolor="#333333", capsize=3, fmt='none', lw=1)
                bar_idx += 1
        if baseline is not None and baseline_label is not None:
            add_baseline(ax, baseline, baseline_label)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"k = {k}")
        ax.set_xlabel("Model")
        ax.set_ylabel(value_col.replace("_", " ").title())
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis('off')
    fig.suptitle(f"{title} — {constructed_by.title()}", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path)
    plt.close(fig)


def model_rows_by_demo(df: pd.DataFrame, value_col: str, weight_col: str, out_path: Path, constructed_by: str, title: str) -> None:
    if df.empty:
        return
    d = df.copy()
    d = d[d["constructed_by"].astype(str).str.lower() == constructed_by]
    if d.empty:
        return
    # Aggregate across frameworks at (model_id, demo pair) and carry denominator
    den_all = d.groupby(["model_id", "demographic_base", "demographic_variant"], dropna=False)[weight_col].sum().rename("den").reset_index()
    d = weighted_mean(d, value_col=value_col, weight_col=weight_col, group_cols=["model_id", "demographic_base", "demographic_variant"]).merge(den_all, on=["model_id", "demographic_base", "demographic_variant"], how="left")
    d["model_short"] = d["model_id"].astype(str).map(short_model_name)
    d["demo_pair"] = d["demographic_base"].astype(str) + " / " + d["demographic_variant"].astype(str)
    # Order models and pairs
    models = d["model_short"].unique().tolist()
    models.sort()
    pairs = sorted(d["demo_pair"].unique().tolist())
    # Compute SE
    d["se"] = np.where(d["den"] > 0, np.sqrt(d[value_col] * (1 - d[value_col]) / d["den"]), np.nan)
    # Figure size: one row per model, width proportional to #pairs
    fig_w = max(28, 0.7 * len(pairs) + 8)
    fig_h = max(8, 5 * len(models))
    fig, axes = plt.subplots(len(models), 1, figsize=(fig_w, fig_h), squeeze=False)
    for i, m in enumerate(models):
        ax = axes[i, 0]
        dm = d[d["model_short"] == m].copy()
        dm = dm.set_index("demo_pair").reindex(pairs).reset_index()  # align pairs across rows
        sns.barplot(data=dm, x="demo_pair", y=value_col, ax=ax, errorbar=None)
        # error bars
        xticks = [t.get_text() for t in ax.get_xticklabels()]
        bar_idx = 0
        for xp in xticks:
            row = dm[dm["demo_pair"] == xp].head(1)
            if not row.empty and pd.notna(row["se"].iloc[0]) and bar_idx < len(ax.patches):
                bar = ax.patches[bar_idx]
                x = bar.get_x() + bar.get_width() / 2
                y = bar.get_height()
                ax.errorbar(x, y, yerr=float(row["se"].iloc[0]), ecolor="#333333", capsize=3, fmt='none', lw=1)
            bar_idx += 1
        ax.set_ylim(0.0, 1.0)
        ax.set_title(m)
        ax.set_xlabel("Demographic Pair")
        ax.set_ylabel(value_col.replace("_", " ").title())
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right")
    fig.suptitle(f"{title} — {constructed_by.title()}", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path)
    plt.close(fig)


def selection_rate_models_rows_symmetrized(df_sr_demo: pd.DataFrame, out_path: Path, constructed_by: str, title: str) -> None:
    if df_sr_demo.empty:
        return
    d = df_sr_demo.copy()
    d = d[d["constructed_by"].astype(str).str.lower() == constructed_by]
    if d.empty:
        return
    # Choose rate and weight columns (prefer nonab)
    rate_col = "selection_rate_A_nonab" if "selection_rate_A_nonab" in d.columns else "selection_rate_A_all"
    w_nonab = "n_nonab" if "n_nonab" in d.columns else None
    w_all = "n_all" if "n_all" in d.columns else None
    d["weight"] = d[w_nonab].where(~d[w_nonab].isna(), d[w_all]).astype(float) if w_nonab and w_all else d[w_all].astype(float)
    d["rateA"] = d[rate_col].where(~d[rate_col].isna(), d.get("selection_rate_A_all", pd.Series(index=d.index, dtype=float))).astype(float)

    # Build unordered pair key
    d["pair_key"] = d.apply(lambda r: " / ".join(sorted([str(r["demographic_base"]), str(r["demographic_variant"])])), axis=1)

    # Helper to compute symmetrized rates per (model_id, pair_key)
    recs = []
    for (model_id, pair), grp in d.groupby(["model_id", "pair_key"], dropna=False):
        # identify groups
        all_groups = set(grp["demographic_base"].astype(str)).union(set(grp["demographic_variant"].astype(str)))
        if len(all_groups) != 2:
            continue
        g1, g2 = sorted(list(all_groups))
        # orientation g1 as base
        r1 = grp[(grp["demographic_base"].astype(str) == g1) & (grp["demographic_variant"].astype(str) == g2)]
        # orientation g2 as base
        r2 = grp[(grp["demographic_base"].astype(str) == g2) & (grp["demographic_variant"].astype(str) == g1)]
        w1 = float(r1["weight"].iloc[0]) if len(r1) else 0.0
        w2 = float(r2["weight"].iloc[0]) if len(r2) else 0.0
        p1 = float(r1["rateA"].iloc[0]) if len(r1) else float("nan")  # P(select g1 when g1 is base)
        p2 = float(r2["rateA"].iloc[0]) if len(r2) else float("nan")  # P(select g2 when g2 is base)
        den = w1 + w2
        if den <= 0:
            continue
        # Symmetrized: P(select g1) = w1*p1 + w2*(1-p2) over total
        p_g1 = (w1 * (0.0 if pd.isna(p1) else p1) + w2 * (0.0 if pd.isna(p2) else (1.0 - p2))) / den
        p_g2 = 1.0 - p_g1
        # record both bars
        recs.append({"model_id": model_id, "pair": pair, "group": g1, "selection_rate": p_g1, "den": den})
        recs.append({"model_id": model_id, "pair": pair, "group": g2, "selection_rate": p_g2, "den": den})

    if not recs:
        return
    s = pd.DataFrame(recs)
    s["model_short"] = s["model_id"].astype(str).map(short_model_name)
    # order models and pairs
    models = sorted(s["model_short"].unique().tolist())
    pairs = sorted(s["pair"].unique().tolist())
    groups = sorted(s["group"].unique().tolist())
    # color mapping per group (consistent across rows)
    palette = sns.color_palette("Set2", n_colors=max(2, len(groups)))
    color_map = {g: palette[i % len(palette)] for i, g in enumerate(groups)}
    # figure size more balanced
    fig_w = max(22, 0.9 * len(pairs) + 8)
    fig_h = max(12, 6 * len(models))
    fig, axes = plt.subplots(len(models), 1, figsize=(fig_w, fig_h), squeeze=False)
    for i, m in enumerate(models):
        ax = axes[i, 0]
        sm = s[s["model_short"] == m].copy()
        # Ensure consistent x ordering
        sm_pairs = pairs
        x = np.arange(len(sm_pairs))
        width = 0.36
        # Compute values for each pair and the two groups present
        # Precompute se per (pair, group)
        sm_se = sm.groupby(["pair", "group"], dropna=False).agg(p=("selection_rate", "first"), den=("den", "first")).reset_index()
        sm_se["se"] = np.where(sm_se["den"] > 0, np.sqrt(sm_se["p"] * (1 - sm_se["p"]) / sm_se["den"]), np.nan)
        # Draw bars pair-by-pair, exactly two bars side-by-side
        for idx, pair_label in enumerate(sm_pairs):
            sub = sm[sm["pair"] == pair_label]
            # pair should have exactly two groups; order alphabetically for stability
            pair_groups = sorted(sub["group"].unique().tolist())
            if len(pair_groups) != 2:
                continue
            gL, gR = pair_groups[0], pair_groups[1]
            # left bar (x - width/2)
            pL = float(sub[sub["group"] == gL]["selection_rate"].iloc[0]) if not sub[sub["group"] == gL].empty else np.nan
            seL = float(sm_se[(sm_se["pair"] == pair_label) & (sm_se["group"] == gL)]["se"].iloc[0]) if not sm_se[(sm_se["pair"] == pair_label) & (sm_se["group"] == gL)].empty else np.nan
            barL = ax.bar(x[idx] - width/2, pL, width=width, color=color_map[gL])
            if not np.isnan(seL):
                ax.errorbar(x[idx] - width/2, pL, yerr=seL, ecolor="#333333", capsize=3, fmt='none', lw=1)
            # right bar (x + width/2)
            pR = float(sub[sub["group"] == gR]["selection_rate"].iloc[0]) if not sub[sub["group"] == gR].empty else np.nan
            seR = float(sm_se[(sm_se["pair"] == pair_label) & (sm_se["group"] == gR)]["se"].iloc[0]) if not sm_se[(sm_se["pair"] == pair_label) & (sm_se["group"] == gR)].empty else np.nan
            barR = ax.bar(x[idx] + width/2, pR, width=width, color=color_map[gR])
            if not np.isnan(seR):
                ax.errorbar(x[idx] + width/2, pR, yerr=seR, ecolor="#333333", capsize=3, fmt='none', lw=1)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(m)
        ax.set_xlabel("Demographic Pair")
        ax.set_ylabel("Selection Rate")
        ax.set_xticks(x)
        ax.set_xticklabels(sm_pairs, rotation=0, ha="center")
        add_baseline(ax, 0.5, "random baseline (1/2)")
        # legend
        handles = [plt.Rectangle((0,0),1,1, color=color_map[g]) for g in groups]
        leg = ax.legend(handles, groups, title="Group", loc="upper right", framealpha=0.4)
        if leg and leg.get_frame():
            leg.get_frame().set_alpha(0.4)
    fig.suptitle(title + f" — {constructed_by.title()}", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path)
    plt.close(fig)


def bars_overall(df: pd.DataFrame, value_col: str, out_path: Path, constructed_by: str, title: str, baseline: float | None = None, baseline_label: str | None = None) -> None:
    if df.empty:
        return
    d = df.copy()
    d = d[d["constructed_by"].astype(str).str.lower() == constructed_by]
    if d.empty:
        return
    d["model_short"] = d["model_id"].astype(str).map(short_model_name)
    d = d.sort_values("model_short")
    fig, ax = plt.subplots(figsize=(16, 7))
    sns.barplot(data=d, x="model_short", y=value_col, ax=ax, errorbar=None)
    # SE from binomial using available denominator: prefer 'den', else 'n', else 'weight'
    denom_col = None
    for c in ["den", "n", "weight"]:
        if c in d.columns:
            denom_col = c
            break
    if denom_col is not None:
        d["se"] = np.where(d[denom_col] > 0, np.sqrt(d[value_col] * (1 - d[value_col]) / d[denom_col]), np.nan)
        xticks = [t.get_text() for t in ax.get_xticklabels()]
        for i, ms in enumerate(xticks):
            row = d[d["model_short"] == ms].head(1)
            if not row.empty and not pd.isna(row["se"].iloc[0]):
                bar = ax.patches[i]
                x = bar.get_x() + bar.get_width() / 2
                y = bar.get_height()
                ax.errorbar(x, y, yerr=float(row["se"].iloc[0]), ecolor="#333333", capsize=3, fmt='none', lw=1)
    if baseline is not None and baseline_label is not None:
        add_baseline(ax, baseline, baseline_label)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Model")
    ax.set_ylabel(value_col.replace("_", " ").title())
    ax.set_title(f"{title} — {constructed_by.title()}")
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    # Load all needed JSONs
    paths = {
        "crit_overall": metrics_dir / "criterion_validity_overall.json",
        "crit_by_k": metrics_dir / "criterion_validity_by_k.json",
        "crit_by_demo": metrics_dir / "criterion_validity_by_demographics.json",
        "uja_overall": metrics_dir / "unjustified_abstention_overall.json",
        "uja_by_k": metrics_dir / "unjustified_abstention_by_k.json",
        "uja_by_demo": metrics_dir / "unjustified_abstention_by_demographics.json",
        "dv_overall": metrics_dir / "discriminant_validity_overall.json",
        "dv_by_k": metrics_dir / "discriminant_validity_by_k.json",
        "dv_by_demo": metrics_dir / "discriminant_validity_by_demographics.json",
        "sr_overall": metrics_dir / "selection_rate_overall.json",
        "sr_by_k": metrics_dir / "selection_rate_by_k.json",
        "sr_by_demo": metrics_dir / "selection_rate_by_demographics.json",
    }
    dfs: Dict[str, pd.DataFrame] = {k: read_json(v) for k, v in paths.items()}
    # Newly added metrics (gender/race selection rate by job)
    sr_gender_job = read_json(metrics_dir / "selection_rate_gender_by_job.json")
    sr_race_job = read_json(metrics_dir / "selection_rate_race_by_job.json")
    # Newly added metrics (criterion validity, unjustified abstention, discriminant validity) by job
    cv_gender_job = read_json(metrics_dir / "criterion_validity_gender_by_job.json")
    cv_race_job = read_json(metrics_dir / "criterion_validity_race3_by_job.json")
    uja_gender_job = read_json(metrics_dir / "unjustified_abstention_gender3_by_job.json")
    uja_race_job = read_json(metrics_dir / "unjustified_abstention_race_by_job.json")
    dv_gender_job = read_json(metrics_dir / "discriminant_validity_gender3_by_job.json")
    dv_race_job = read_json(metrics_dir / "discriminant_validity_race3_by_job.json")
    cv_gender_job = read_json(metrics_dir / "criterion_validity_gender3_by_job.json")

    # For each constructed_by dataset separately
    for constructed in ["claude", "gemini"]:
        # Criterion validity
        bars_by_k(
            dfs["crit_by_k"].rename(columns={"k": "k"}),
            value_col="criterion_validity",
            weight_col="n",
            out_path=out_dir / f"criterion_validity_vs_k_bars_{constructed}.png",
            constructed_by=constructed,
            title="Criterion Validity vs k",
            baseline=1/3,
            baseline_label="random baseline (1/3)",
        )
        model_rows_by_demo(
            dfs["crit_by_demo"],
            value_col="criterion_validity",
            weight_col="n",
            out_path=out_dir / f"criterion_validity_by_demographics_models_rows_{constructed}.png",
            constructed_by=constructed,
            title="Criterion Validity by Demographics",
        )
        # overall with error bars: compute denominators
        crit_den = dfs["crit_overall"].groupby(["model_id", "constructed_by"], dropna=False)["n"].sum().rename("den").reset_index()
        crit_over = weighted_mean(dfs["crit_overall"], value_col="criterion_validity", weight_col="n", group_cols=["model_id", "constructed_by"]).merge(crit_den, on=["model_id", "constructed_by"], how="left")
        bars_overall(
            crit_over,
            value_col="criterion_validity",
            out_path=out_dir / f"criterion_validity_overall_bars_{constructed}.png",
            constructed_by=constructed,
            title="Criterion Validity Overall",
            baseline=1/3,
            baseline_label="random baseline (1/3)",
        )

        # Unjustified abstention
        bars_by_k(
            dfs["uja_by_k"].rename(columns={"k": "k"}),
            value_col="unjustified_abstention",
            weight_col="n",
            out_path=out_dir / f"unjustified_abstention_vs_k_bars_{constructed}.png",
            constructed_by=constructed,
            title="Unjustified Abstention vs k",
        )
        model_rows_by_demo(
            dfs["uja_by_demo"],
            value_col="unjustified_abstention",
            weight_col="n",
            out_path=out_dir / f"unjustified_abstention_by_demographics_models_rows_{constructed}.png",
            constructed_by=constructed,
            title="Unjustified Abstention by Demographics",
        )
        uja_den = dfs["uja_overall"].groupby(["model_id", "constructed_by"], dropna=False)["n"].sum().rename("den").reset_index()
        uja_over = weighted_mean(dfs["uja_overall"], value_col="unjustified_abstention", weight_col="n", group_cols=["model_id", "constructed_by"]).merge(uja_den, on=["model_id", "constructed_by"], how="left")
        bars_overall(
            uja_over,
            value_col="unjustified_abstention",
            out_path=out_dir / f"unjustified_abstention_overall_bars_{constructed}.png",
            constructed_by=constructed,
            title="Unjustified Abstention Overall",
        )

        # Discriminant validity
        bars_by_k(
            dfs["dv_by_k"].rename(columns={"k": "k"}),
            value_col="discriminant_validity",
            weight_col="n",
            out_path=out_dir / f"discriminant_validity_vs_k_bars_{constructed}.png",
            constructed_by=constructed,
            title="Discriminant Validity vs k",
        )
        model_rows_by_demo(
            dfs["dv_by_demo"],
            value_col="discriminant_validity",
            weight_col="n",
            out_path=out_dir / f"discriminant_validity_by_demographics_models_rows_{constructed}.png",
            constructed_by=constructed,
            title="Discriminant Validity by Demographics",
        )
        dv_den = dfs["dv_overall"].groupby(["model_id", "constructed_by"], dropna=False)["n"].sum().rename("den").reset_index()
        dv_over = weighted_mean(dfs["dv_overall"], value_col="discriminant_validity", weight_col="n", group_cols=["model_id", "constructed_by"]).merge(dv_den, on=["model_id", "constructed_by"], how="left")
        bars_overall(
            dv_over,
            value_col="discriminant_validity",
            out_path=out_dir / f"discriminant_validity_overall_bars_{constructed}.png",
            constructed_by=constructed,
            title="Discriminant Validity Overall",
        )

        # Selection rate (tie-breaking)
        # Prefer non-abstain rate when available; set appropriate weights
        srk = dfs["sr_by_k"].copy()
        if not srk.empty:
            srk["value"] = srk["selection_rate_A_nonab"].where(~srk["selection_rate_A_nonab"].isna(), srk["selection_rate_A_all"]).astype(float)
            srk["w"] = np.where(~srk["selection_rate_A_nonab"].isna(), srk["n_nonab"], srk["n_all"]).astype(float)
            srk = srk.rename(columns={"value": "selection_rate", "w": "weight", "k": "k"})
        bars_by_k(
            srk,
            value_col="selection_rate",
            weight_col="weight",
            out_path=out_dir / f"selection_rate_vs_k_bars_{constructed}.png",
            constructed_by=constructed,
            title="Selection Rate (base/first candidate chosen) vs k",
            baseline=0.5,
            baseline_label="random baseline (1/2)",
        )
        srd = dfs["sr_by_demo"].copy()
        if not srd.empty:
            selection_rate_models_rows_symmetrized(
                srd,
                out_path=out_dir / f"selection_rate_by_demographics_models_rows_{constructed}.png",
                constructed_by=constructed,
                title="Selection Rate by Demographic Group (symmetrized)",
            )
        sro_raw = dfs["sr_overall"].copy()
        sro_final = pd.DataFrame(columns=["model_id", "constructed_by", "selection_rate", "den"])  # default empty
        if not sro_raw.empty:
            sro_raw["selection_rate"] = sro_raw["selection_rate_A_nonab"].where(~sro_raw["selection_rate_A_nonab"].isna(), sro_raw["selection_rate_A_all"]).astype(float)
            sro_raw["weight"] = np.where(~sro_raw["selection_rate_A_nonab"].isna(), sro_raw["n_nonab"], sro_raw["n_all"]).astype(float)
            sr_den = sro_raw.groupby(["model_id", "constructed_by"], dropna=False)["weight"].sum().rename("den").reset_index()
            sro_final = weighted_mean(sro_raw, value_col="selection_rate", weight_col="weight", group_cols=["model_id", "constructed_by"]).merge(sr_den, on=["model_id", "constructed_by"], how="left")
        bars_overall(
            sro_final,
            value_col="selection_rate",
            out_path=out_dir / f"selection_rate_overall_bars_{constructed}.png",
            constructed_by=constructed,
            title="Selection Rate (base/first resume chosen) Overall",
            baseline=0.5,
            baseline_label="random baseline (1/2)",
        )

        # -----------------------------
        # Gender vs Women selection by top 5 jobs
        # -----------------------------
        def bars_by_job_models(d: pd.DataFrame, jobs: List[str], title: str, outfile: Path, value_col: str, baseline: float | None = None, baseline_label: str | None = None, ylabel: str | None = None, group_col: str = "group") -> None:
            if d.empty:
                return
            dd = d.copy()
            dd = dd[dd["constructed_by"].astype(str).str.lower() == constructed]
            if dd.empty:
                return
            # Determine group column
            if group_col not in dd.columns:
                if "pair" in dd.columns:
                    group_col = "pair"
                else:
                    return
            # filter to top 5 jobs (exact names provided)
            top_jobs = jobs
            dd = dd[dd["job_title"].isin(top_jobs)].copy()
            if dd.empty:
                return
            # compute SE
            dd["model_short"] = dd["model_id"].astype(str).map(short_model_name)
            dd["pval"] = pd.to_numeric(dd[value_col], errors="coerce")
            dd["se"] = np.where(dd["den"] > 0, np.sqrt(dd["pval"] * (1 - dd["pval"]) / dd["den"]), np.nan)
            # figure: one row per job
            n = len(top_jobs)
            fig_w = max(18, 2.0 * len(dd["model_short"].unique()) + 8)
            fig_h = max(12, 5 * n)
            fig, axes = plt.subplots(n, 1, figsize=(fig_w, fig_h), squeeze=False)
            for i, job in enumerate(top_jobs):
                ax = axes[i, 0]
                dj = dd[dd["job_title"] == job].copy()
                if dj.empty:
                    ax.axis('off')
                    continue
                # barplot with hue=group (two bars per model)
                order_models = sorted(dj["model_short"].unique().tolist())
                hue_order = sorted(dj[group_col].unique().tolist())
                sns.barplot(data=dj, x="model_short", y=value_col, hue=group_col, hue_order=hue_order, order=order_models, ax=ax, errorbar=None)
                # add error bars
                xticks = [t.get_text() for t in ax.get_xticklabels()]
                # seaborn groups patches by model×hue in order
                patch_idx = 0
                for ms in order_models:
                    for grp in hue_order:
                        row = dj[(dj["model_short"] == ms) & (dj[group_col] == grp)].head(1)
                        if not row.empty and patch_idx < len(ax.patches):
                            se_val = float(row["se"].iloc[0]) if pd.notna(row["se"].iloc[0]) else np.nan
                            if not np.isnan(se_val):
                                bar = ax.patches[patch_idx]
                                x = bar.get_x() + bar.get_width() / 2
                                y = bar.get_height()
                                ax.errorbar(x, y, yerr=se_val, ecolor="#333333", capsize=3, fmt='none', lw=1)
                        patch_idx += 1
                ax.set_ylim(0.0, 1.0)
                ax.set_title(job)
                ax.set_xlabel("Model")
                ax.set_ylabel(ylabel or value_col.replace("_", " ").title())
                plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
                if baseline is not None and baseline_label is not None:
                    add_baseline(ax, baseline, baseline_label)
                leg = ax.legend(title="Group", loc="upper right", framealpha=0.4)
                if leg and leg.get_frame():
                    leg.get_frame().set_alpha(0.4)
            fig.suptitle(title + f" — {constructed.title()}", fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(outfile)
            plt.close(fig)

        top5_jobs = ["Product Designer", "Solutions Architect", "Product Manager", "Software Engineer", "Solutions Engineer"]
        # Gender: groups should be M and W
        bars_by_job_models(
            sr_gender_job,
            top5_jobs,
            title="Selection Rate by Gender (Men vs Women) — Top Jobs",
            outfile=out_dir / f"selection_rate_gender_by_job_{constructed}.png",
            value_col="selection_rate",
            baseline=0.5,
            baseline_label="random baseline (1/2)",
            ylabel="Selection Rate",
        )
        # Race: groups should be B and W
        bars_by_job_models(
            sr_race_job,
            top5_jobs,
            title="Selection Rate by Race (Black vs White) — Top Jobs",
            outfile=out_dir / f"selection_rate_race_by_job_{constructed}.png",
            value_col="selection_rate",
            baseline=0.5,
            baseline_label="random baseline (1/2)",
            ylabel="Selection Rate",
        )
        # Criterion Validity by Gender & Race — Top Jobs
        bars_by_job_models(
            cv_gender_job,
            top5_jobs,
            title="Criterion Validity by Gender — Top Jobs",
            outfile=out_dir / f"criterion_validity_gender_by_job_{constructed}.png",
            value_col="rate",
            baseline=None,
            baseline_label=None,
            ylabel="Criterion Validity",
        )
        bars_by_job_models(
            cv_race_job,
            top5_jobs,
            title="Criterion Validity by Race — Top Jobs",
            outfile=out_dir / f"criterion_validity_race_by_job_{constructed}.png",
            value_col="rate",
            baseline=None,
            baseline_label=None,
            ylabel="Criterion Validity",
        )
        # Unjustified Abstention by Gender & Race — Top Jobs
        bars_by_job_models(
            uja_gender_job,
            top5_jobs,
            title="Unjustified Abstention by Gender — Top Jobs",
            outfile=out_dir / f"unjustified_abstention_gender_by_job_{constructed}.png",
            value_col="rate",
            baseline=None,
            baseline_label=None,
            ylabel="Unjustified Abstention",
        )
        bars_by_job_models(
            uja_race_job,
            top5_jobs,
            title="Unjustified Abstention by Race — Top Jobs",
            outfile=out_dir / f"unjustified_abstention_race_by_job_{constructed}.png",
            value_col="rate",
            baseline=None,
            baseline_label=None,
            ylabel="Unjustified Abstention",
        )
        # Discriminant Validity by Gender & Race — Top Jobs
        bars_by_job_models(
            dv_gender_job,
            top5_jobs,
            title="Discriminant Validity by Gender — Top Jobs",
            outfile=out_dir / f"discriminant_validity_gender_by_job_{constructed}.png",
            value_col="rate",
            baseline=None,
            baseline_label=None,
            ylabel="Discriminant Validity",
        )
        bars_by_job_models(
            dv_race_job,
            top5_jobs,
            title="Discriminant Validity by Race — Top Jobs",
            outfile=out_dir / f"discriminant_validity_race_by_job_{constructed}.png",
            value_col="rate",
            baseline=None,
            baseline_label=None,
            ylabel="Discriminant Validity",
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[FATAL] Unhandled exception in plot_brainstormed_metrics")
        print(traceback.format_exc())
        sys.exit(1)


