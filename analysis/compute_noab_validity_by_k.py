#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import numpy as np


def short_model_name_fallback(m: str) -> str:
    return str(m)


def main() -> None:
    base_dir = Path("/home/zs7353/resume_validity/evaluations/baseline_noabstain")
    patterns = ["*_exp_validity.csv", "*_exp_validity_demographics.csv"]

    try:
        sys.path.append("/home/zs7353/resume_validity/analysis")
        import plot_aggregates_rel6 as pa  # type: ignore
        short_model = pa.short_model_name  # type: ignore
    except Exception:
        short_model = short_model_name_fallback

    all_rows: list[pd.DataFrame] = []
    for pat in patterns:
        for fp in base_dir.glob(pat):
            try:
                df = pd.read_csv(fp)
            except Exception as e:
                print(f"[WARN] failed to load {fp}: {e}")
                continue
            if df.empty:
                continue
            # Attach model id derived from filename prefix
            df["model_id"] = fp.name.split("_")[0]
            all_rows.append(df)

    if not all_rows:
        print("No forced-decision validity files found under baseline_noabstain")
        return

    df = pd.concat(all_rows, ignore_index=True)

    # Keep only validity experiments
    if "experiment_type" in df.columns:
        df = df[df["experiment_type"].isin(["validity", "validity_demographics"])].copy()

    # Keep k ∈ {1,2,3}
    df["num_differed"] = pd.to_numeric(df["num_differed"], errors="coerce")
    df = df[df["num_differed"].isin([1, 2, 3])].copy()

    # Only preferred / underqualified pairs; ignore reworded/equal
    df["pair_type"] = df["pair_type"].astype(str).str.lower()
    df = df[df["pair_type"].isin(["preferred", "underqualified"])].copy()

    # Normalize decisions
    df["decision"] = df["decision"].astype(str).str.lower()
    df = df[df["decision"].isin(["first", "second"])].copy()

    # Canonical groups set for same/cross
    allowed = {"B_W", "B_M", "W_W", "W_M"}

    # df_k: same-demographic canonical groups (for k table)
    if {"demographic_base", "demographic_variant"}.issubset(df.columns):
        df_k = df[
            (df["demographic_base"].astype(str) == df["demographic_variant"].astype(str))
            & (df["demographic_base"].isin(allowed))
        ].copy()
    else:
        df_k = df.copy()

    if df_k.empty:
        print("No eligible rows after filtering (check that validity runs completed)")
        return

    # Compute correctness per row based on pair_type
    is_under = df_k["pair_type"].eq("underqualified") & df_k["decision"].eq("first")
    is_pref = df_k["pair_type"].eq("preferred") & df_k["decision"].eq("second")
    df_k["is_correct"] = is_under | is_pref

    agg = (
        df_k.groupby(["model_id", "num_differed"], dropna=False)
        .agg(n=("is_correct", "size"), k=("is_correct", "sum"))
        .reset_index()
    )
    agg["accuracy"] = np.where(agg["n"] > 0, agg["k"] / agg["n"], np.nan)
    agg["se"] = np.sqrt(agg["accuracy"] * (1 - agg["accuracy"]) / agg["n"].replace(0, np.nan))
    agg["model_short"] = agg["model_id"].map(short_model)

    # Pivot tables
    pivot_acc = agg.pivot_table(index="model_short", columns="num_differed", values="accuracy", aggfunc="first").sort_index()
    pivot_n = agg.pivot_table(index="model_short", columns="num_differed", values="n", aggfunc="first").sort_index()
    pivot_se = agg.pivot_table(index="model_short", columns="num_differed", values="se", aggfunc="first").sort_index()

    # Pretty print
    pd.set_option("display.max_rows", None)
    print("Forced-decision validity by model × k (accuracy):")
    if not pivot_acc.empty:
        print(pivot_acc.round(3).to_string())
    else:
        print("<empty>")

    print("\nCounts (n) by model × k:")
    if not pivot_n.empty:
        print(pivot_n.to_string())
    else:
        print("<empty>")

    print("\nStd. Error by model × k:")
    if not pivot_se.empty:
        print(pivot_se.round(3).to_string())
    else:
        print("<empty>")

    # LaTeX table for model × k accuracy (same-demo canonical already enforced above)
    if not pivot_acc.empty:
        ks = [int(k) for k in pivot_acc.columns]
        print("\n=== LaTeX: Forced-decision Validity by k (Same-Demo) ===")
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\caption{Forced-decision validity (same-demographics) by number of differed qualifications}")
        print("\\label{tab:noab_validity_by_k}")
        print("\\begin{tabular}{l" + "c" * len(ks) + "}")
        print("\\toprule")
        header = "Model" + "".join([f" & k={k}" for k in ks]) + " \\\\"  # newline
        print(header)
        print("\\midrule")
        for model_short, row in pivot_acc.iterrows():
            vals = [f"{row.get(k, np.nan):.3f}" if pd.notna(row.get(k, np.nan)) else "--" for k in ks]
            print(model_short.replace("_", "\\_") + " " + "".join([f"& {v} " for v in vals]) + "\\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")

    # Same vs Cross demographics overall (k pooled within 1..3)
    # Create same/cross subsets from df (k in 1..3, preferred/underqualified, decisions forced)
    same_mask = (
        (df["demographic_base"].astype(str) == df["demographic_variant"].astype(str))
        & df["demographic_base"].isin(allowed)
    ) if {"demographic_base", "demographic_variant"}.issubset(df.columns) else pd.Series([False] * len(df))
    cross_mask = (
        (df["demographic_base"].astype(str) != df["demographic_variant"].astype(str))
        & df["demographic_base"].isin(allowed)
        & df["demographic_variant"].isin(allowed)
    ) if {"demographic_base", "demographic_variant"}.issubset(df.columns) else pd.Series([False] * len(df))

    def summarize(mask: pd.Series) -> pd.DataFrame:
        d = df[mask].copy()
        if d.empty:
            return pd.DataFrame(columns=["model_id", "n", "k", "accuracy"])
        is_under = d["pair_type"].eq("underqualified") & d["decision"].eq("first")
        is_pref = d["pair_type"].eq("preferred") & d["decision"].eq("second")
        d["is_correct"] = is_under | is_pref
        g = d.groupby("model_id").agg(n=("is_correct", "size"), k=("is_correct", "sum")).reset_index()
        g["accuracy"] = np.where(g["n"] > 0, g["k"] / g["n"], np.nan)
        g["model_short"] = g["model_id"].map(short_model)
        return g.sort_values("model_short")

    same_sum = summarize(same_mask)
    cross_sum = summarize(cross_mask)

    # Merge to one table
    merged = pd.merge(
        same_sum[["model_short", "accuracy", "n"]].rename(columns={"accuracy": "same_acc", "n": "same_n"}),
        cross_sum[["model_short", "accuracy", "n"]].rename(columns={"accuracy": "cross_acc", "n": "cross_n"}),
        on="model_short",
        how="outer",
    ).fillna({"same_acc": np.nan, "cross_acc": np.nan, "same_n": 0, "cross_n": 0}).sort_values("model_short")

    if not merged.empty:
        print("\nSame vs Cross-demographics forced-decision accuracy (k=1..3 pooled):")
        print(merged[["model_short", "same_acc", "same_n", "cross_acc", "cross_n"]].assign(
            same_acc=lambda d: d["same_acc"].round(3),
            cross_acc=lambda d: d["cross_acc"].round(3),
        ).to_string(index=False))

        # LaTeX
        print("\n=== LaTeX: Same vs Cross Demographics (Forced-decision) ===")
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\caption{Forced-decision validity by demographics (same vs cross), k=1..3 pooled}")
        print("\\label{tab:noab_same_vs_cross}")
        print("\\begin{tabular}{lcc}")
        print("\\toprule")
        print("Model & Same-demo Acc & Cross-demo Acc \\\")
        print("\\midrule")
        for _, r in merged.iterrows():
            ms = str(r["model_short"]).replace("_", "\\_")
            sa = "--" if pd.isna(r["same_acc"]) else f"{r['same_acc']:.3f}"
            ca = "--" if pd.isna(r["cross_acc"]) else f"{r['cross_acc']:.3f}"
            print(f"{ms} & {sa} & {ca} \\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")


if __name__ == "__main__":
    main()


