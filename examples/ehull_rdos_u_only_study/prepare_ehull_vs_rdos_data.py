#!/usr/bin/env python3
"""
Prepare data for E_hull vs 2.5*r_DOS gnuplot figure.

Reads:
  - high_throughput_mace_results.full.csv (E_hull values)
  - doscar_rewards.csv (r_DOS values)
  - all_compounds_by_composite_score.csv (top 10 MCTS predictions)

Outputs .dat files for gnuplot in gnuplot_data_ehull_rdos/
"""

import pandas as pd
import re
from pathlib import Path


# Known synthesized compounds (experimentally verified)
SYNTHESIZED = {'V6Sn6U', 'Cr6Ge6U', 'Co6Ge6U', 'Nb6Sn6U'}

# Known unsynthesized compounds (in target structural family but not synthesized)
UNSYNTHESIZED = {
    'Ti6Ge6U', 'Ti6Sn6U', 'V6Ge6U', 'Mn6Ge6U', 'Mn6Sn6U', 'Fe6Sn6U',
    'Ni6Ge6U', 'Ni6Sn6U', 'Cu6Ge6U', 'Cu6Sn6U', 'Pd6Sn6U', 'Ag6Sn6U',
    'Hf6Si6U', 'Hf6Ge6U', 'Ta6Sn6U', 'Nb6Ge6U'
}

# Element categories for formula parsing
GROUP_IV = {'Si', 'Ge', 'Sn', 'Pb'}
F_BLOCK = {
    'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
    'Tm', 'Yb', 'Lu', 'Th', 'Pa', 'U', 'Np', 'Pu'
}
TRANSITION_METALS = {
    'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'
}


def parse_mcts_formula(formula):
    """Parse MCTS formula (e.g. V6Sn6U) into (f_block, group_iv, tm) elements."""
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)

    f_elem, g_iv_elem, tm_elem = None, None, None
    for elem, count in matches:
        if not elem:
            continue
        if elem in F_BLOCK:
            f_elem = elem
        elif elem in GROUP_IV:
            g_iv_elem = elem
        elif elem in TRANSITION_METALS:
            tm_elem = elem

    return f_elem, g_iv_elem, tm_elem


def mcts_to_doscar(formula):
    """Convert MCTS formula (e.g. V6Sn6U) to DOSCAR format (e.g. U-Sn-V)."""
    f_elem, g_iv_elem, tm_elem = parse_mcts_formula(formula)
    if f_elem and g_iv_elem and tm_elem:
        return f"{f_elem}-{g_iv_elem}-{tm_elem}"
    return None


def main():
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent

    # Read MACE results
    mace_csv = base_dir / 'high_throughput_mace_results.full.csv'
    df_mace = pd.read_csv(mace_csv)
    print(f"Loaded {len(df_mace)} compounds from MACE CSV")

    # Filter to U-containing, no Ce
    def has_element(name, elem):
        """Check if formula contains a specific element (not as substring of another)."""
        pattern = r'(?<![a-z])' + elem + r'(?![a-z])'
        return bool(re.search(pattern, name))

    df_u = df_mace[df_mace['name'].apply(lambda x: has_element(x, 'U'))].copy()
    df_u = df_u[~df_u['name'].apply(lambda x: has_element(x, 'Ce'))].copy()
    print(f"U-containing (no Ce): {len(df_u)} compounds")

    # Read DOSCAR rewards
    doscar_csv = base_dir / 'doscar_rewards.csv'
    df_doscar = pd.read_csv(doscar_csv)
    doscar_dict = dict(zip(df_doscar['compound_name'], df_doscar['reward_normalized']))
    print(f"Loaded {len(doscar_dict)} DOSCAR rewards")

    # Join: add r_DOS to MACE data
    df_u['doscar_name'] = df_u['name'].apply(mcts_to_doscar)
    df_u['r_dos'] = df_u['doscar_name'].map(doscar_dict).fillna(0.0)

    matched = df_u[df_u['r_dos'] > 0]
    print(f"Matched compounds (r_DOS > 0): {len(matched)} of {len(df_u)}")

    # Create output directory
    output_dir = script_dir / 'gnuplot_data_ehull_rdos'
    output_dir.mkdir(exist_ok=True)

    # 1. All U compounds (background grey dots)
    all_file = output_dir / 'all_u_compounds.dat'
    with open(all_file, 'w') as f:
        f.write("# e_hull  2.5*r_dos  name\n")
        for _, row in df_u.iterrows():
            f.write(f"{row['e_above_hull']}\t{2.5 * row['r_dos']}\t{row['name']}\n")
    print(f"Created {all_file} ({len(df_u)} compounds)")

    # 2. Synthesized compounds
    synth_file = output_dir / 'synthesized_compounds.dat'
    synth_count = 0
    with open(synth_file, 'w') as f:
        f.write("# e_hull  2.5*r_dos  name\n")
        for _, row in df_u.iterrows():
            if row['name'] in SYNTHESIZED:
                f.write(f"{row['e_above_hull']}\t{2.5 * row['r_dos']}\t{row['name']}\n")
                synth_count += 1
    print(f"Created {synth_file} ({synth_count} compounds)")

    # 3. Unsynthesized compounds
    unsynth_file = output_dir / 'unsynthesized_compounds.dat'
    unsynth_count = 0
    with open(unsynth_file, 'w') as f:
        f.write("# e_hull  2.5*r_dos  name\n")
        for _, row in df_u.iterrows():
            if row['name'] in UNSYNTHESIZED:
                f.write(f"{row['e_above_hull']}\t{2.5 * row['r_dos']}\t{row['name']}\n")
                unsynth_count += 1
    print(f"Created {unsynth_file} ({unsynth_count} compounds)")

    # 4. Top 10 MCTS predictions from this run
    composite_csv = script_dir / 'all_compounds_by_composite_score.csv'
    top10_file = output_dir / 'top10_mcts.dat'
    if composite_csv.exists():
        df_composite = pd.read_csv(composite_csv)
        top10 = df_composite.head(10)

        # Look up r_DOS for top 10 (use values from MACE+DOSCAR join)
        mace_lookup = dict(zip(df_u['name'], zip(df_u['e_above_hull'], df_u['r_dos'])))

        with open(top10_file, 'w') as f:
            f.write("# e_hull  2.5*r_dos  name\n")
            for _, row in top10.iterrows():
                name = row['name']
                if name in mace_lookup:
                    e_hull, r_dos = mace_lookup[name]
                    f.write(f"{e_hull}\t{2.5 * r_dos}\t{name}\n")
                else:
                    # Fallback to composite CSV values
                    f.write(f"{row['e_above_hull']}\t{2.5 * row['r_DOS']}\t{name}\n")
        print(f"Created {top10_file} (10 compounds)")
    else:
        print(f"Warning: {composite_csv} not found")

    # Print summary
    print(f"\nData files ready in {output_dir}/")
    print(f"  all_u_compounds.dat     - {len(df_u)} compounds (background)")
    print(f"  synthesized_compounds.dat    - {synth_count} compounds")
    print(f"  unsynthesized_compounds.dat  - {unsynth_count} compounds")
    print(f"  top10_mcts.dat          - top 10 MCTS predictions")

    # Print some stats
    print(f"\nE_hull range: [{df_u['e_above_hull'].min():.4f}, {df_u['e_above_hull'].max():.4f}]")
    print(f"2.5*r_DOS range: [{2.5*df_u['r_dos'].min():.4f}, {2.5*df_u['r_dos'].max():.4f}]")

    return 0


if __name__ == '__main__':
    exit(main())
