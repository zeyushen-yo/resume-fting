#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import sys


def short_model_name_fallback(m: str) -> str:
    return str(m)


def main() -> None:
    base = Path('/home/zs7353/resume_validity/evaluations/baseline_noabstain')
    patterns = ['*_exp_validity.csv', '*_exp_validity_demographics.csv']

    try:
        sys.path.append('/home/zs7353/resume_validity/analysis')
        import plot_aggregates_rel6 as pa  # type: ignore
        short = pa.short_model_name  # type: ignore
    except Exception:
        short = short_model_name_fallback

    rows = []
    for pat in patterns:
        for fp in base.glob(pat):
            try:
                df = pd.read_csv(fp)
            except Exception as e:
                print(f"[WARN] {fp.name}: {e}")
                continue
            if df.empty:
                continue
            df['model_id'] = fp.name.split('_')[0]
            rows.append(df)

    if not rows:
        print('No forced-decision validity files found')
        return

    df = pd.concat(rows, ignore_index=True)
    if 'experiment_type' in df.columns:
        df = df[df['experiment_type'].isin(['validity','validity_demographics'])]

    # Keep forced choices and relevant pair types
    df['pair_type'] = df['pair_type'].astype(str).str.lower()
    df = df[df['pair_type'].isin(['preferred','underqualified'])]
    df['decision'] = df['decision'].astype(str).str.lower()
    df = df[df['decision'].isin(['first','second'])]

    # k ∈ {1,2,3}
    df['num_differed'] = pd.to_numeric(df['num_differed'], errors='coerce')
    df = df[df['num_differed'].isin([1,2,3])].copy()

    # Cross demographics among canonical groups
    allowed = {'B_W','B_M','W_W','W_M'}
    if {'demographic_base','demographic_variant'}.issubset(df.columns):
        cross_df = df[(df['demographic_base']!=df['demographic_variant']) & df['demographic_base'].isin(allowed) & df['demographic_variant'].isin(allowed)].copy()
    else:
        cross_df = pd.DataFrame(columns=df.columns)

    if cross_df.empty:
        print('No cross-demographic rows after filtering')
        return

    cross_df['is_correct'] = ((cross_df['pair_type']=='underqualified') & (cross_df['decision']=='first')) | ((cross_df['pair_type']=='preferred') & (cross_df['decision']=='second'))
    agg = cross_df.groupby(['model_id','num_differed']).agg(n=('is_correct','size'), k=('is_correct','sum')).reset_index()
    agg['accuracy'] = np.where(agg['n']>0, agg['k']/agg['n'], np.nan)
    agg['model_short'] = agg['model_id'].map(short)
    agg = agg.sort_values(['model_short','num_differed'])

    # Plain text table
    pivot_acc = agg.pivot_table(index='model_short', columns='num_differed', values='accuracy', aggfunc='first').sort_index()
    print('Cross-demographic forced-decision validity by model × k (accuracy):')
    if not pivot_acc.empty:
        print(pivot_acc.round(3).to_string())
    else:
        print('<empty>')

    # LaTeX table
    ks = sorted(agg['num_differed'].unique().tolist())
    models = sorted(agg['model_short'].unique().tolist())
    lines = []
    lines.append('\\begin{table}[h]')
    lines.append('\\centering')
    lines.append('\\caption{Forced-decision validity (cross-demographics) by number of differed qualifications}')
    lines.append('\\label{tab:noab_validity_by_k_cross}')
    lines.append('\\begin{tabular}{l' + 'c'*len(ks) + '}')
    lines.append('\\toprule')
    lines.append('Model' + ''.join([f' & k={int(k)}' for k in ks]) + ' \\\\')
    lines.append('\\midrule')
    for m in models:
        row = agg[agg['model_short']==m].set_index('num_differed')
        vals = []
        for k in ks:
            v = row.at[k,'accuracy'] if (k in row.index and pd.notna(row.at[k,'accuracy'])) else np.nan
            vals.append('--' if pd.isna(v) else f"{v:.3f}")
        lines.append(m.replace('_','\\_') + ' ' + ''.join([f'& {v} ' for v in vals]) + '\\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    print('\n=== LaTeX: Cross-demographics Forced-decision Validity by k ===')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()


