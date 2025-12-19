#!/usr/bin/env python3
"""
Recompute metrics after filtering out error pairs identified in the blacklist.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Set

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser("Recompute metrics with error pairs filtered out")
    p.add_argument("--blacklist", type=str, required=True, help="JSON file with error pair hashes")
    p.add_argument("--eval_dirs", nargs="+", required=True, help="Evaluation directories to process")
    p.add_argument("--out_metrics", type=str, required=True, help="Output CSV for core metrics")
    p.add_argument("--out_gender", type=str, required=True, help="Output CSV for gender selection rates")
    p.add_argument("--out_race", type=str, required=True, help="Output CSV for race selection rates")
    return p.parse_args()


def pair_hash(base_resume: str, variant_resume: str) -> str:
    """Create a unique hash for a resume pair."""
    combined = base_resume + "|||" + variant_resume
    return hashlib.md5(combined.encode('utf-8')).hexdigest()


def load_blacklist(path: Path) -> Set[str]:
    """Load set of error pair hashes."""
    with open(path, 'r') as f:
        return set(json.load(f))


def load_and_filter_evals(eval_dirs, blacklist: Set[str]):
    """Load all evaluation CSVs and filter out blacklisted pairs."""
    frames = []
    for eval_dir in eval_dirs:
        p = Path(eval_dir)
        if not p.is_dir():
            continue
        for fp in sorted(p.glob("*.csv")):
            try:
                df = pd.read_csv(fp)
            except:
                continue
            if df.empty:
                continue
            
            # Compute hash for each row
            hashes = []
            for _, row in df.iterrows():
                base = str(row.get('base_resume', ''))
                variant = str(row.get('variant_resume', ''))
                h = pair_hash(base, variant)
                hashes.append(h)
            df['pair_hash'] = hashes
            
            # Filter out blacklisted
            before = len(df)
            df = df[~df['pair_hash'].isin(blacklist)].copy()
            after = len(df)
            if before > after:
                print(f"  {fp.name}: filtered {before - after}/{before} pairs")
            
            # Add metadata
            df['source_file'] = fp.name
            df['framework'] = 'baseline_noabstain' if 'noabstain' in str(fp) else 'baseline'
            df['model_id'] = fp.name.split('_paired_resume')[0]
            frames.append(df)
    
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compute_core_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute CV, UJA, DV, SR per job/model."""
    results = []
    
    for (job_title, model_id), g in df.groupby(['job_title', 'model_id']):
        # Strict pairs (num_differed > 0) from baseline
        g_baseline = g[g['framework'] == 'baseline']
        strict = g_baseline[g_baseline['num_differed'].fillna(0) > 0]
        n_strict = len(strict)
        cv = float(strict['is_valid'].sum()) / n_strict if n_strict > 0 else float('nan')
        uja = float(strict['abstained'].sum()) / n_strict if n_strict > 0 else float('nan')
        
        # Equal pairs (num_differed == 0) from baseline
        equal = g_baseline[g_baseline['num_differed'].fillna(0) == 0]
        n_equal = len(equal)
        dv = float(equal['abstained'].sum()) / n_equal if n_equal > 0 else float('nan')
        
        # Equal pairs from no-abstain
        g_noab = g[g['framework'] == 'baseline_noabstain']
        equal_noab = g_noab[g_noab['num_differed'].fillna(0) == 0]
        n_equal_noab = len(equal_noab)
        sr = float((equal_noab['decision'].str.lower() == 'first').sum()) / n_equal_noab if n_equal_noab > 0 else float('nan')
        
        results.append({
            'job_title': job_title,
            'model_id': model_id,
            'criterion_validity': cv,
            'unjustified_abstention': uja,
            'discriminant_validity': dv,
            'selection_rate_first': sr,
            'n_strict_pairs': n_strict,
            'n_equal_pairs': n_equal,
            'n_equal_pairs_noab': n_equal_noab,
        })
    
    return pd.DataFrame(results)


def compute_selection_by_group(df: pd.DataFrame, group_mapper, col_name: str):
    """Compute symmetrized selection rates by demographic group."""
    # Only no-abstain equal pairs
    e = df[(df['framework'] == 'baseline_noabstain') & (df['num_differed'].fillna(0) == 0)].copy()
    if e.empty:
        return pd.DataFrame()
    
    records = []
    for (job_title, model_id), g in e.groupby(['job_title', 'model_id']):
        # Symmetrize
        counts = {}
        for _, r in g.iterrows():
            base = r['demographic_base']
            var = r['demographic_variant']
            d = str(r['decision']).lower()
            
            # Only count valid decisions
            if d not in ('first', 'second'):
                continue
            
            for code in (base, var):
                counts.setdefault(code, {'sel': 0.0, 'tot': 0.0})
                counts[code]['tot'] += 1.0
            
            if d == 'first':
                counts[base]['sel'] += 1.0
            elif d == 'second':
                counts[var]['sel'] += 1.0
        
        # Map to dimension and aggregate
        dim_counts = {}
        for code, st in counts.items():
            dim_val = group_mapper(code)
            if dim_val:
                dim_counts.setdefault(dim_val, {'sel': 0.0, 'tot': 0.0})
                dim_counts[dim_val]['sel'] += st['sel']
                dim_counts[dim_val]['tot'] += st['tot']
        
        for dim_val, st in dim_counts.items():
            if st['tot'] > 0:
                records.append({
                    'job_title': job_title,
                    'model_id': model_id,
                    'group': dim_val,
                    'selection_rate': st['sel'] / st['tot'],
                    'den': st['tot'],
                })
    
    return pd.DataFrame(records)


def map_gender(code: str):
    try:
        _, g = code.split('_', 1)
        return {'M': 'Men', 'W': 'Women'}.get(g)
    except:
        return None


def map_race(code: str):
    try:
        r, _ = code.split('_', 1)
        return {'B': 'Black', 'W': 'White'}.get(r)
    except:
        return None


def main():
    args = parse_args()
    
    # Load blacklist
    blacklist = load_blacklist(Path(args.blacklist))
    print(f"Loaded blacklist with {len(blacklist)} error pairs")
    
    # Load and filter evaluations
    print("\nLoading and filtering evaluation results...")
    df = load_and_filter_evals(args.eval_dirs, blacklist)
    print(f"Total rows after filtering: {len(df)}")
    
    # Compute core metrics
    print("\nComputing core metrics...")
    metrics = compute_core_metrics(df)
    metrics.to_csv(args.out_metrics, index=False, quoting=csv.QUOTE_ALL)
    print(f"Wrote {len(metrics)} rows -> {args.out_metrics}")
    
    # Compute selection by gender
    print("\nComputing selection rates by gender...")
    gender = compute_selection_by_group(df, map_gender, 'gender')
    gender.to_csv(args.out_gender, index=False)
    print(f"Wrote {len(gender)} rows -> {args.out_gender}")
    
    # Compute selection by race
    print("\nComputing selection rates by race...")
    race = compute_selection_by_group(df, map_race, 'race')
    race.to_csv(args.out_race, index=False)
    print(f"Wrote {len(race)} rows -> {args.out_race}")


if __name__ == "__main__":
    main()


