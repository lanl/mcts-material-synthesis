#!/usr/bin/env python3
"""
Prepare data for top 10 grouped vertical bar chart (E_hull and 2.5*r_DOS).
Sorted by composite score (same order as top10_by_modified_composite.pdf).
"""

import pandas as pd
from pathlib import Path
from ase.formula import Formula


def is_target_compound(formula: str) -> bool:
    """Check if compound is one of the target compounds."""
    targets = ['V6Sn6U', 'Nb6Sn6U', 'Cr6Ge6U', 'Co6Ge6U']
    if formula in targets:
        return True
    for target in targets:
        try:
            f1 = Formula(formula)
            f2 = Formula(target)
            if f1.count() == f2.count():
                return True
        except:
            pass
    return False


def main():
    script_dir = Path(__file__).parent

    csv_file = script_dir / 'all_compounds_by_composite_score.csv'
    if not csv_file.exists():
        print(f"Error: {csv_file} not found")
        return 1

    df = pd.read_csv(csv_file)
    top10 = df.head(10)

    output_dir = script_dir / 'gnuplot_data_bars'
    output_dir.mkdir(exist_ok=True)

    # Data file: rank name e_above_hull 2.5*r_DOS is_target
    data_file = output_dir / 'top10_ehull_rdos_bars.dat'
    with open(data_file, 'w') as f:
        f.write("# rank  e_hull  weighted_rdos  is_target  name\n")
        for i, (idx, row) in enumerate(top10.iterrows(), 1):
            is_target = 1 if is_target_compound(row['name']) else 0
            weighted_rdos = 2.5 * row['r_DOS']
            f.write(f"{i}\t{row['e_above_hull']:.6f}\t{weighted_rdos:.6f}\t{is_target}\t{row['name']}\n")

    print(f"Created {data_file}")

    # Labels file for x-axis
    labels_file = output_dir / 'top10_bar_labels.dat'
    with open(labels_file, 'w') as f:
        for i, (idx, row) in enumerate(top10.iterrows(), 1):
            is_target = 1 if is_target_compound(row['name']) else 0
            f.write(f"{i} {row['name']} {is_target}\n")

    print(f"Created {labels_file}")

    # Print summary
    print(f"\nTop 10 compounds (sorted by composite score):")
    for i, (idx, row) in enumerate(top10.iterrows(), 1):
        target = " *" if is_target_compound(row['name']) else ""
        print(f"  {i:2d}. {row['name']:12s}  E_hull={row['e_above_hull']:8.4f}  "
              f"2.5*r_DOS={2.5*row['r_DOS']:.4f}  composite={row['composite_score']:.4f}{target}")

    return 0


if __name__ == '__main__':
    exit(main())
