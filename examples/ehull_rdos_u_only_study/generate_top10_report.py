#!/usr/bin/env python3
"""
Generate top 10 compounds report for the ehull_rdos reward.

Reads all_compounds.csv from MCTS output and computes:
- ehull_reward: tanh-transformed energy above hull reward (-tanh(300*(e_hull-0.05)))
- composite_score: beta*ehull_reward + gamma*r_DOS

Weights for this study: beta=1.0, gamma=2.5 (E_form is tracked for reference only,
it is not part of the reward - see mcts_crystal/node.py:ehull_reward)
"""

import sys
import pandas as pd
from pathlib import Path
from ase.formula import Formula

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mcts_crystal.node import ehull_reward


def main():
    script_dir = Path(__file__).parent

    # Read all_compounds.csv
    csv_file = script_dir / "all_compounds.csv"
    if not csv_file.exists():
        print(f"Error: all_compounds.csv not found in {script_dir}")
        return 1

    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} compounds from {csv_file}")

    # Rename columns for consistency
    df['name'] = df['formula']
    df['r_DOS'] = df['dos_reward']

    # Compute the E_hull reward
    df['ehull_reward'] = df['e_above_hull'].apply(ehull_reward)

    beta = 1.0
    gamma = 2.5

    df['weighted_r_DOS'] = gamma * df['r_DOS']
    df['composite_score'] = beta * df['ehull_reward'] + df['weighted_r_DOS']

    # Sort by composite score (descending)
    df_sorted = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
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

    def write_report(path, rows, title):
        with open(path, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write(f"{title}\n")
            f.write("=" * 100 + "\n")
            f.write(f"\nComposite Score = beta*ehull_reward + gamma*r_DOS\n")
            f.write(f"Weights: beta={beta}, gamma={gamma}\n")
            f.write(f"E_form is tracked for reference only and is not part of the reward.\n")
            f.write(f"\nTotal compounds explored: {len(df)}\n")
            f.write("\n" + "=" * 100 + "\n\n")
            for i, (_, row) in enumerate(rows.iterrows(), 1):
                f.write(f"Rank {i}: {row['name']}\n")
                f.write(f"  E_form (reference only): {row['e_form']:8.4f} eV/atom\n")
                f.write(f"  E_hull:                  {row['e_above_hull']:8.4f} eV/atom\n")
                f.write(f"  ehull_reward:            {row['ehull_reward']:8.4f}\n")
                f.write(f"  r_DOS:                   {row['r_DOS']:8.4f}\n")
                f.write(f"  weighted_r_DOS:          {row['weighted_r_DOS']:8.4f}  (gamma * r_DOS)\n")
                f.write(f"  composite_score:         {row['composite_score']:8.4f}\n")
                f.write("\n")
        print(f"Report saved to: {path}")

    write_report(script_dir / "all_compounds_by_composite_score.txt", df_sorted, "ALL COMPOUNDS RANKED BY COMPOSITE SCORE")
    write_report(script_dir / "top10_compounds_by_composite_score.txt", top10, "TOP 10 COMPOUNDS BY COMPOSITE SCORE")

    # Check target compounds
    target_compounds = ['V6Sn6U', 'Nb6Sn6U', 'Cr6Ge6U', 'Co6Ge6U']
    print("\n" + "=" * 100)
    print("TARGET COMPOUND DISCOVERY CHECK")
    print("=" * 100)
    for compound in target_compounds:
        found = False
        stored_as = None
        if compound in df_sorted['name'].values:
            found = True
            stored_as = compound
        else:
            for formula_str in df_sorted['name'].values:
                try:
                    f1 = Formula(compound)
                    f2 = Formula(formula_str)
                    if f1.count() == f2.count():
                        found = True
                        stored_as = formula_str
                        break
                except Exception:
                    pass
        if found:
            rank = df_sorted[df_sorted['name'] == stored_as].index[0] + 1
            row = df_sorted[df_sorted['name'] == stored_as].iloc[0]
            print(f"  {compound:12s} - Rank {rank:3d}  composite={row['composite_score']:7.4f}")
        else:
            print(f"  {compound:12s} - NOT DISCOVERED")

    # Console summary
    print("\n" + "=" * 100)
    print(f"TOP 10 COMPOUNDS BY COMPOSITE SCORE (beta={beta}, gamma={gamma})")
    print("=" * 100)
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        print(f"{i:2d}. {row['name']:15s}  composite={row['composite_score']:7.4f}  "
              f"(E_hull={row['e_above_hull']:7.4f}, ehull_reward={row['ehull_reward']:6.3f}, "
              f"DOS={row['r_DOS']:5.3f}, wDOS={row['weighted_r_DOS']:5.3f})")

    return 0


if __name__ == '__main__':
    sys.exit(main())
