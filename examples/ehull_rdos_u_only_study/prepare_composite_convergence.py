#!/usr/bin/env python3
"""
Compute best composite reward as a function of iteration.

Reads convergence_history.csv and all_compounds_by_composite_score.csv
to reconstruct which compounds were discovered at each iteration,
then tracks the running best composite score.
"""

import pandas as pd
from pathlib import Path


def main():
    script_dir = Path(__file__).parent

    # Load composite scores for all compounds
    composite_csv = script_dir / 'all_compounds_by_composite_score.csv'
    if not composite_csv.exists():
        print(f"Error: {composite_csv} not found — run generate_top10_report.py first")
        return 1

    df_comp = pd.read_csv(composite_csv)
    composite_lookup = dict(zip(df_comp['name'], df_comp['composite_score']))
    print(f"Loaded composite scores for {len(composite_lookup)} compounds")

    # Load convergence history
    conv_csv = script_dir / 'convergence_history.csv'
    if not conv_csv.exists():
        print(f"Error: {conv_csv} not found")
        return 1

    df_conv = pd.read_csv(conv_csv)
    print(f"Loaded convergence history: {len(df_conv)} iterations")

    # Track all formulas seen across iterations from the "best" columns
    formulas_seen = set()
    best_composite_history = []

    for _, row in df_conv.iterrows():
        # Collect formulas from all "best" tracking columns
        for col in ['best_e_form_formula', 'best_e_hull_formula', 'best_rdos_formula']:
            if col in df_conv.columns and pd.notna(row[col]):
                formulas_seen.add(row[col])

        # Compute best composite among all formulas seen so far
        best = float('-inf')
        for f in formulas_seen:
            if f in composite_lookup:
                best = max(best, composite_lookup[f])

        if best == float('-inf'):
            best = 0.0

        best_composite_history.append(best)

    # Write .dat file for gnuplot
    dat_file = script_dir / 'composite_convergence.dat'
    with open(dat_file, 'w') as f:
        f.write("# iteration\tbest_composite\n")
        for i, comp in enumerate(best_composite_history):
            f.write(f"{i}\t{comp:.6f}\n")

    print(f"Created {dat_file} ({len(best_composite_history)} rows)")
    print(f"  Final best composite: {best_composite_history[-1]:.4f}")
    print(f"  Unique formulas tracked: {len(formulas_seen)}")

    return 0


if __name__ == '__main__':
    exit(main())
