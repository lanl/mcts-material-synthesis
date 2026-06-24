#!/usr/bin/env python3
"""
Generate top 10 compounds report for the ehull_rdos reward.

Reads all_compounds.csv from MCTS output and computes:
- ehull_reward: tanh-transformed energy above hull reward (-tanh(120*(e_hull-0.05)))
- composite_score: beta*ehull_reward + gamma*r_DOS

Weights for this study: beta=1.0, gamma=0.0001 (E_form is tracked for reference only,
it is not part of the reward - see mcts_crystal/node.py:ehull_reward). gamma is loaded
from config.json so it stays in sync with the value used during the MCTS run.
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mcts_crystal.node import ehull_reward
from mcts_crystal.cli import load_config
from mcts_crystal.doscar_utils import DoscarRewardLookup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthesized_compounds import SYNTHESIZED_COMPOUNDS


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
    # Compute r_DOS: prefer existing `dos_reward` if available, otherwise compute
    # in real time from the raw peaks file (no precomputed rewards cache)
    if 'dos_reward' in df.columns:
        df['r_DOS'] = df['dos_reward']
    else:
        # search up to a few parent levels for the peaks file
        MAX_PARENT_DEPTH = 4
        search_roots = [script_dir] + list(script_dir.parents)[:MAX_PARENT_DEPTH]
        peaks_path = None
        for r in search_roots:
            try:
                cand = list(Path(r).rglob('doscar_peaks_data_with_U.csv'))
            except OSError:
                cand = []
            if cand:
                peaks_path = cand[0]
                break
        df['r_DOS'] = 0.0
        if peaks_path is not None:
            try:
                dos_dict = DoscarRewardLookup(peaks_file=str(peaks_path)).rewards_dict

                F_BLOCK = {'Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Th','Pa','U','Np','Pu','Ac'}
                import re
                def parse_elems(s):
                    if pd.isna(s):
                        return []
                    if '-' in str(s):
                        parts = [p for p in re.split('[^A-Za-z]', str(s)) if p]
                        return parts
                    return re.findall(r'[A-Z][a-z]?', str(s))

                dos_by_key = {}
                for name, val in dos_dict.items():
                    try:
                        v = float(val)
                    except Exception:
                        continue
                    elems = parse_elems(name)
                    key = tuple(sorted([e for e in elems if e not in F_BLOCK]))
                    if not key:
                        continue
                    dos_by_key[key] = max(dos_by_key.get(key, float('-inf')), v)
                def key_from_name(name):
                    elems = parse_elems(name)
                    return tuple(sorted([e for e in elems if e not in F_BLOCK]))
                for idx, row in df.iterrows():
                    key = key_from_name(row['name'])
                    if key in dos_by_key:
                        df.at[idx, 'r_DOS'] = dos_by_key[key]
            except Exception:
                pass

    # Compute the E_hull reward
    df['ehull_reward'] = df['e_above_hull'].apply(ehull_reward)

    config = load_config(str(Path(__file__).resolve().parents[2] / 'config.json'))
    beta = float(config.get('beta', 1.0))
    gamma = float(config.get('gamma', 0.0001))

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

    # --- Top 15 by Combined Reward (stability + rDOS), dash-name format ---
    # f(rDOS) and f(E_hull) below are the same r_DOS/ehull_reward columns
    # computed above; Total is the same composite_score (beta*ehull_reward +
    # gamma*r_DOS, gamma loaded from config.json - NOT an unweighted sum).
    _name_lookup = DoscarRewardLookup(peaks_file="/nonexistent.csv")  # only used for name conversion, no data needed
    df_sorted['dash_name'] = df_sorted['name'].apply(_name_lookup.convert_formula_to_doscar_format)
    df_sorted['dash_name'] = df_sorted['dash_name'].fillna(df_sorted['name'])
    df_sorted['is_synthesized'] = df_sorted['dash_name'].isin(SYNTHESIZED_COMPOUNDS)
    df_sorted['has_tc'] = df_sorted['dash_name'].str.contains(r'(?:^|-)Tc(?:-|$)', regex=True)

    def write_combined_reward_table(path, ranked, title):
        # f(rDOS) is the *weighted* contribution (gamma * r_DOS), so that
        # f(rDOS) + f(E_hull) == Total (composite_score) exactly - consistent
        # with f(E_hull) already being the E_hull term's contribution to Total.
        lines = []
        lines.append(f"   {title} (* = Priority Match):")
        lines.append(f"   {'Rank':<6} {'Compound':<22} {'f(rDOS)':>8}  {'E_hull':>8}  {'f(E_hull)':>9}  {'Total':>8}")
        lines.append(f"   {'----':<6} {'-'*22} {'-'*8}  {'-'*8}  {'-'*9}  {'-'*8}")
        for i, (_, row) in enumerate(ranked.head(15).iterrows(), 1):
            marker = '*' if row['is_synthesized'] else ' '
            lines.append(
                f"   {i:>3}. {marker} {row['dash_name']:<22} {row['weighted_r_DOS']:8.4f}  "
                f"{row['e_above_hull']:8.4f}  {row['ehull_reward']:9.4f}  {row['composite_score']:8.4f}"
            )
        lines.append("")
        lines.append(f"   Target compound rankings ({title}):")
        for compound in SYNTHESIZED_COMPOUNDS:
            match = ranked[ranked['dash_name'] == compound]
            if match.empty:
                lines.append(f"   -> {compound:15s}  NOT DISCOVERED")
                continue
            row = match.iloc[0]
            rank = ranked.index.get_loc(match.index[0]) + 1
            lines.append(
                f"   -> {compound:15s}  rank={rank:3d}  f(rDOS)={row['weighted_r_DOS']:.4f}  "
                f"f(E_hull)={row['ehull_reward']:.4f}  total={row['composite_score']:.4f}"
            )

        text = "\n".join(lines)
        print("\n" + text)
        with open(path, 'w') as f:
            f.write(text + "\n")
        print(f"\nReport saved to: {path}")

    all_ranked = df_sorted.sort_values('composite_score', ascending=False).reset_index(drop=True)
    write_combined_reward_table(
        script_dir / "top15_combined_reward_all.txt", all_ranked,
        "Top 15 All U-Compounds by Combined Reward"
    )

    no_tc_ranked = df_sorted[~df_sorted['has_tc']].sort_values('composite_score', ascending=False).reset_index(drop=True)
    write_combined_reward_table(
        script_dir / "top15_combined_reward_no_tc.txt", no_tc_ranked,
        "Top 15 U-Compounds (excluding Tc) by Combined Reward"
    )

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
