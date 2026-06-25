#!/usr/bin/env python3
"""
Generate top 10 compounds report for the ehull_rdos reward.

Delegates scoring to generate_figures.compute_composite() - the canonical
beta*ehull_reward + gamma*r_DOS implementation, with r_DOS recomputed from
doscar_peaks_data_with_U.csv rather than the (possibly stale) dos_reward
column logged during the MCTS run - so this report's ranking always matches
tables/top15_u_only.tex and the other generated figures.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_figures import compute_composite, load_gamma


def main():
    script_dir = Path(__file__).parent

    csv_file = script_dir / "all_compounds.csv"
    if not csv_file.exists():
        print(f"Error: all_compounds.csv not found in {script_dir}")
        return 1

    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} compounds from {csv_file}")

    gamma = load_gamma()
    df_sorted = compute_composite(df, beta=1.0, gamma=gamma)
    top10 = df_sorted.head(10)

    columns = ['name', 'e_form', 'e_above_hull', 'ehull_reward', 'r_DOS', 'weighted_r_DOS', 'composite_score']

    # Save ALL compounds as CSV
    csv_output_all = script_dir / "all_compounds_by_composite_score.csv"
    df_sorted[columns].to_csv(csv_output_all, index=False)
    print(f"All compounds CSV saved to: {csv_output_all}")

    # Save top 10 as CSV
    csv_output_top10 = script_dir / "top10_compounds_by_composite_score.csv"
    top10[columns].to_csv(csv_output_top10, index=False)
    print(f"Top 10 compounds CSV saved to: {csv_output_top10}")

    # Console summary
    print("\n" + "=" * 100)
    print(f"TOP 10 COMPOUNDS BY COMPOSITE SCORE (beta=1.0, gamma={gamma:g})")
    print("=" * 100)
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        print(f"{i:2d}. {row['name']:15s}  composite={row['composite_score']:7.4f}  "
              f"(E_hull={row['e_above_hull']:7.4f}, ehull_reward={row['ehull_reward']:6.3f}, "
              f"DOS={row['r_DOS']:5.3f}, wDOS={row['weighted_r_DOS']:5.3f})")

    return 0


if __name__ == '__main__':
    sys.exit(main())
