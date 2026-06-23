#!/usr/bin/env python3
"""
Prepare data for gnuplot visualization of top 10 compounds from the ehull_rdos study.
Reads all_compounds_by_composite_score.csv and creates .dat files for plotting.
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
        except Exception:
            pass

    return False


def main():
    csv_file = Path('all_compounds_by_composite_score.csv')

    if not csv_file.exists():
        print(f"Error: {csv_file} not found")
        return 1

    df = pd.read_csv(csv_file)
    top10 = df.head(10)

    print(f"Top 10 compounds from {csv_file}:")
    print(top10[['name', 'ehull_reward', 'r_DOS', 'composite_score']])

    output_dir = Path('gnuplot_data')
    output_dir.mkdir(exist_ok=True)

    # Labels file: rank formula is_target
    labels_file = output_dir / 'top10_labels.dat'
    with open(labels_file, 'w') as f:
        for i, (idx, row) in enumerate(top10.iterrows(), 1):
            is_target = 1 if is_target_compound(row['name']) else 0
            f.write(f"{i} {row['name']} {is_target}\n")
    print(f"\nCreated {labels_file}")

    # Components file: rank ehull_reward r_DOS is_target
    components_file = output_dir / 'top10_components.dat'
    with open(components_file, 'w') as f:
        f.write("# rank ehull_reward r_DOS is_target\n")
        for i, (idx, row) in enumerate(top10.iterrows(), 1):
            is_target = 1 if is_target_compound(row['name']) else 0
            f.write(f"{i} {row['ehull_reward']:.6f} {row['r_DOS']:.6f} {is_target}\n")
    print(f"Created {components_file}")

    # Composite scores file: rank composite_score
    composite_file = output_dir / 'top10_composite.dat'
    with open(composite_file, 'w') as f:
        f.write("# rank composite_score\n")
        for i, (idx, row) in enumerate(top10.iterrows(), 1):
            f.write(f"{i} {row['composite_score']:.6f}\n")
    print(f"Created {composite_file}")

    # Print target compounds found
    targets_found = []
    for i, (idx, row) in enumerate(top10.iterrows(), 1):
        if is_target_compound(row['name']):
            targets_found.append((i, row['name']))

    print(f"\nTarget compounds in top 10: {len(targets_found)}")
    for rank, name in targets_found:
        print(f"  Rank {rank}: {name}")

    print(f"\nData files ready for gnuplot in {output_dir}/")
    return 0


if __name__ == '__main__':
    exit(main())
