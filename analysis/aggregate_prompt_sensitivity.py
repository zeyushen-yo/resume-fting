#!/usr/bin/env python3
"""
Aggregate prompt sensitivity evaluation results into metrics CSVs.
Produces separate outputs for each prompt variant (human, llm).
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import List
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Aggregate prompt sensitivity results")
    p.add_argument(
        "--input_dir",
        type=str,
        default="/scratch/gpfs/KOROLOVA/zs7353/resume_validity/evaluations/prompt_sensitivity",
        help="Directory containing evaluation CSVs",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="/scratch/gpfs/KOROLOVA/zs7353/resume_validity/evaluations/prompt_sensitivity/aggregated",
        help="Directory to write aggregated CSVs",
    )
    return p.parse_args()


def load_all_csvs(input_dir: Path) -> pd.DataFrame:
    """Load and concatenate all CSV files from the input directory."""
    all_dfs = []
    for csv_file in sorted(input_dir.glob("*.csv")):
        if csv_file.name.startswith("aggregated") or "metrics" in csv_file.name:
            continue
        try:
            df = pd.read_csv(csv_file)
            all_dfs.append(df)
            print(f"Loaded {len(df)} rows from {csv_file.name}")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if not all_dfs:
        raise ValueError("No CSV files found")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal rows loaded: {len(combined)}")
    return combined


def clean_model_name(model: str) -> str:
    """Extract clean model name from full path."""
    if "/" in model:
        return model.split("/")[1]
    return model


def compute_metrics(df: pd.DataFrame, prompt_variant: str) -> pd.DataFrame:
    """Compute validity metrics for a specific prompt variant."""
    # Filter to this prompt variant
    df_variant = df[df["prompt_variant"] == prompt_variant].copy()
    
    if df_variant.empty:
        print(f"No data for prompt variant: {prompt_variant}")
        return pd.DataFrame()
    
    # Normalize is_valid column
    df_variant["is_valid"] = df_variant["is_valid"].astype(str).str.lower().isin(["true", "1", "yes"])
    
    # Normalize pair_type
    df_variant["pair_type"] = df_variant["pair_type"].astype(str).str.lower()
    
    # Clean model names
    df_variant["model_clean"] = df_variant["model"].apply(clean_model_name)
    
    # Filter to validity experiment type only (excluding fairness)
    df_validity = df_variant[df_variant["experiment_type"].astype(str).str.lower() == "validity"].copy()
    
    # For validity metrics, we focus on underqualified and preferred pairs (not reworded/equal)
    # "Strict" pairs are underqualified or preferred (where there IS a clear better candidate)
    df_strict = df_validity[df_validity["pair_type"].isin(["underqualified", "preferred"])].copy()
    df_equal = df_validity[df_validity["pair_type"] == "reworded"].copy()
    
    results = []
    
    for model in sorted(df_variant["model_clean"].unique()):
        model_strict = df_strict[df_strict["model_clean"] == model]
        model_equal = df_equal[df_equal["model_clean"] == model]
        
        n_strict = len(model_strict)
        n_equal = len(model_equal)
        
        # Criterion validity: accuracy on strict pairs (did model pick the correct better resume?)
        if n_strict > 0:
            criterion_validity = model_strict["is_valid"].mean()
            # Unjustified abstention: abstained when there was a clear answer
            abstained_col = model_strict["abstained"].astype(str).str.lower().isin(["true", "1", "yes"])
            unjustified_abstention = abstained_col.mean()
        else:
            criterion_validity = 0.0
            unjustified_abstention = 0.0
        
        # Discriminant validity: on equal pairs, did model abstain (correct) or pick one (incorrect)?
        # For equal pairs, abstaining is correct behavior
        if n_equal > 0:
            abstained_equal = model_equal["abstained"].astype(str).str.lower().isin(["true", "1", "yes"])
            discriminant_validity = abstained_equal.mean()
            n_equal_noab = (~abstained_equal).sum()
        else:
            discriminant_validity = 0.0
            n_equal_noab = 0
        
        # Selection rate for first resume (bias indicator)
        model_all = df_validity[df_validity["model_clean"] == model]
        if len(model_all) > 0:
            first_selected = model_all["decision"].astype(str).str.lower() == "first"
            selection_rate_first = first_selected.mean()
        else:
            selection_rate_first = 0.0
        
        results.append({
            "model_id": model,
            "prompt_variant": prompt_variant,
            "criterion_validity": criterion_validity,
            "unjustified_abstention": unjustified_abstention,
            "discriminant_validity": discriminant_validity,
            "selection_rate_first": selection_rate_first,
            "n_strict_pairs": n_strict,
            "n_equal_pairs": n_equal,
            "n_equal_pairs_noab": n_equal_noab,
        })
    
    return pd.DataFrame(results)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all data
    df = load_all_csvs(input_dir)
    
    # Check what prompt variants we have
    if "prompt_variant" not in df.columns:
        print("ERROR: 'prompt_variant' column not found in data")
        return
    
    variants = df["prompt_variant"].unique()
    print(f"\nFound prompt variants: {list(variants)}")
    
    # Aggregate for each variant
    all_metrics = []
    for variant in variants:
        print(f"\n=== Processing variant: {variant} ===")
        metrics = compute_metrics(df, variant)
        if not metrics.empty:
            # Save per-variant file
            out_file = output_dir / f"prompt_sensitivity_{variant}_metrics.csv"
            metrics.to_csv(out_file, index=False, quoting=csv.QUOTE_ALL)
            print(f"Saved: {out_file}")
            all_metrics.append(metrics)
    
    # Also save combined file
    if all_metrics:
        combined = pd.concat(all_metrics, ignore_index=True)
        combined_file = output_dir / "prompt_sensitivity_all_metrics.csv"
        combined.to_csv(combined_file, index=False, quoting=csv.QUOTE_ALL)
        print(f"\nSaved combined: {combined_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(combined.to_string(index=False))


if __name__ == "__main__":
    main()

