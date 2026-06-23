#!/usr/bin/env python3
"""
Generate publication figures as PNGs directly from MCTS outputs and high-throughput data.

This script replaces the prior gnuplot/prepare pipeline and writes PNG files:
- top10_by_composite.png
- top10_ehull_rdos_bars.png
- ehull_vs_rdos.png
- composite_convergence.png
- radial_tree_composite.png (delegates to create_composite_radial_tree.py)

It expects to be run from `analysis/ehull_rdos_u_only_study/` where the MCTS
run outputs (`all_compounds.csv`, `convergence_history.csv`, `mcts_object.pkl`) are
present (the example `run_study.sh` places them there). It also expects the
high-throughput cache `high_throughput_mace_results.full.csv` and
`doscar_rewards.csv` to be available at the repository root for the
`ehull_vs_rdos` figure if the analysis CSV is not present.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess
import sys


def compute_composite(df, beta=1.0, gamma=2.5):
    try:
        from mcts_crystal.node import ehull_reward
    except Exception:
        # Fallback: define local ehull_reward matching mcts_crystal.node implementation
        def ehull_reward(e_hull: float) -> float:
            import numpy as _np
            return -_np.tanh(300.0 * (e_hull - 0.05))
    df = df.copy()
    df['name'] = df.get('formula', df.get('name'))
    # normalize available DOS column names
    if 'dos_reward' in df.columns and 'r_DOS' not in df.columns:
        df['r_DOS'] = df['dos_reward']
    df['r_DOS'] = df.get('r_DOS', 0.0).fillna(0.0)
    df['ehull_reward'] = df['e_above_hull'].apply(ehull_reward)
    df['weighted_r_DOS'] = gamma * df['r_DOS']
    df['composite_score'] = beta * df['ehull_reward'] + df['weighted_r_DOS']
    df_sorted = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    return df_sorted


def plot_top10(df_sorted, out_dir: Path):
    top10 = df_sorted.head(10)

    # Horizontal stacked bar (composite components)
    fig, ax = plt.subplots(figsize=(4, 4))
    ranks = np.arange(1, len(top10) + 1)
    ehull_vals = top10['ehull_reward'].values
    rdos_vals = top10['r_DOS'].values * 2.5

    ax.barh(ranks, ehull_vals, color="#ff7f0e", edgecolor='k')
    ax.barh(ranks, rdos_vals, left=ehull_vals, color="#2ca02c", edgecolor='k')
    ax.set_yticks(ranks)
    ax.set_yticklabels(top10['name'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Composite components')
    plt.tight_layout()
    out = out_dir / 'top10_by_composite.png'
    fig.savefig(out, dpi=300)
    plt.close(fig)

    # Grouped vertical bars: E_hull and 2.5*r_DOS
    fig, ax = plt.subplots(figsize=(4, 4))
    x = np.arange(len(top10))
    width = 0.35
    ax.bar(x - width/2, top10['e_above_hull'].values, width, label='E_hull', color="#ff7f0e")
    ax.bar(x + width/2, 2.5 * top10['r_DOS'].values, width, label='2.5*r_DOS', color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(top10['name'].values, rotation=45, ha='right')
    ax.set_ylabel('E_hull / 2.5*r_DOS')
    ax.legend()
    plt.tight_layout()
    out = out_dir / 'top10_ehull_rdos_bars.png'
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_ehull_vs_rdos(repo_root: Path, out_dir: Path):
    # Prefer using the analysis CSV that already contains r_DOS and e_above_hull
    comp_csv = out_dir / 'all_compounds_by_composite_score.csv'
    if comp_csv.exists():
        df_all = pd.read_csv(comp_csv)
        # Filter U-containing (no Ce) by formula/name field
        def has_element(name, elem):
            import re
            pattern = r'(?<![a-z])' + elem + r'(?![a-z])'
            return bool(re.search(pattern, str(name)))

        if 'name' not in df_all.columns and 'formula' in df_all.columns:
            df_all['name'] = df_all['formula']

        df_u = df_all[df_all['name'].apply(lambda x: has_element(x, 'U'))].copy()
        df_u = df_u[~df_u['name'].apply(lambda x: has_element(x, 'Ce'))].copy()
        # Ensure r_DOS column exists (case-insensitive fallback)
        if 'r_DOS' not in df_u.columns and 'dos_reward' in df_u.columns:
            df_u['r_DOS'] = df_u['dos_reward']
        df_u['r_dos'] = df_u['r_DOS'].fillna(0.0)
    else:
        mace_csv = repo_root / 'high_throughput_mace_results.full.csv'
        doscar_csv = repo_root / 'doscar_rewards.csv'
        if not mace_csv.exists() or not doscar_csv.exists():
            print('Skipping ehull_vs_rdos: missing data sources')
            return

        df_mace = pd.read_csv(mace_csv)
        df_dos = pd.read_csv(doscar_csv)
        # build name->reward mapping from available columns
        if 'compound_name' in df_dos.columns and 'reward_normalized' in df_dos.columns:
            dos_dict = dict(zip(df_dos['compound_name'], df_dos['reward_normalized']))
        else:
            # fallback: try name/reward
            dos_dict = dict(zip(df_dos.iloc[:, 0], df_dos.iloc[:, 1]))

        def _norm(s):
            if pd.isna(s):
                return ''
            return ''.join([c for c in str(s).lower() if c.isalnum()])

        dos_dict_norm = { _norm(k): v for k, v in dos_dict.items() }
        df_mace['name_norm'] = df_mace.get('name', df_mace.get('formula')).apply(_norm)
        df_mace['r_dos'] = df_mace['name_norm'].map(dos_dict_norm).fillna(0.0)
        if 'name' not in df_mace.columns and 'formula' in df_mace.columns:
            df_mace['name'] = df_mace['formula']
        df_u = df_mace[df_mace['name'].apply(lambda x: 'u' in str(x).lower())].copy()
        df_u = df_u[~df_u['name'].apply(lambda x: 'ce' in str(x).lower())].copy()

    # Some entries may not map; attempt conversion using same heuristic as DoscarRewardLookup
    # For robustness, keep 0.0 where not found.

    fig, ax = plt.subplots(figsize=(4, 4))
    # Use r_dos (already present) or fall back to r_DOS column
    rdos_vals = df_u.get('r_dos', df_u.get('r_DOS', pd.Series(0.0))).astype(float)
    ax.scatter(2.5 * rdos_vals, df_u['e_above_hull'].astype(float), s=6, color='#D0D0D0', label='All Compounds')

    # Top10 overlay if available (look in out_dir)
    composite_csv = out_dir / 'all_compounds_by_composite_score.csv'
    if composite_csv.exists():
        df_comp = pd.read_csv(composite_csv)
        top10 = df_comp.head(10)
        mace_lookup = dict(zip(df_u['name'], zip(df_u['e_above_hull'], df_u.get('r_dos', df_u.get('r_DOS', pd.Series(0.0))))))
        xs, ys, labels = [], [], []
        for _, row in top10.iterrows():
            name = row['name'] if 'name' in row else row.get('formula')
            if name in mace_lookup:
                e_hull, rdos = mace_lookup[name]
                xs.append(2.5 * rdos)
                ys.append(e_hull)
                labels.append(name)
        ax.scatter(xs, ys, s=30, color='#17BECF', label='Top 10 Predictions')

    ax.set_xlabel('2.5 * r_DOS')
    ax.set_ylabel('E_hull (eV/atom)')
    ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out = out_dir / 'ehull_vs_rdos.png'
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_composite_convergence(out_dir: Path):
    # Load convergence history and composite CSV
    conv_csv = out_dir / 'convergence_history.csv'
    composite_csv = out_dir / 'all_compounds_by_composite_score.csv'
    if not conv_csv.exists() or not composite_csv.exists():
        print('Skipping composite_convergence: missing files')
        return

    df_conv = pd.read_csv(conv_csv)
    df_comp = pd.read_csv(composite_csv)
    composite_lookup = dict(zip(df_comp['name'], df_comp['composite_score']))

    formulas_seen = set()
    best_composite_history = []
    best_ehull_reward_history = []
    best_weighted_rdos_history = []
    best_rdos_max_history = []

    gamma = 2.5

    for _, row in df_conv.iterrows():
        for col in ['best_e_form_formula', 'best_e_hull_formula', 'best_rdos_formula']:
            if col in df_conv.columns and pd.notna(row[col]):
                formulas_seen.add(row[col])

        # Determine current best composite and corresponding components
        best_comp = float('-inf')
        best_ehull = np.nan
        best_rdos = np.nan
        # track max rDOS among all seen formulas (monotonic)
        current_max_rdos = float('-inf')

        for f in formulas_seen:
            if f in composite_lookup:
                comp = composite_lookup[f]
                if comp > best_comp:
                    best_comp = comp
                    # Try to extract component values from df_comp
                    try:
                        rowf = df_comp[df_comp['name'] == f].iloc[0]
                        best_ehull = rowf.get('ehull_reward', np.nan)
                        best_rdos = rowf.get('r_DOS', np.nan)
                    except Exception:
                        best_ehull = np.nan
                        best_rdos = np.nan
            # also compute max rDOS among all seen formulas
            try:
                rowf2 = df_comp[df_comp['name'] == f].iloc[0]
                rtmp = rowf2.get('r_DOS', np.nan)
                if pd.notna(rtmp):
                    current_max_rdos = max(current_max_rdos, float(rtmp))
            except Exception:
                pass

        if best_comp == float('-inf'):
            best_comp = 0.0
        if current_max_rdos == float('-inf'):
            current_max_rdos = 0.0

        best_composite_history.append(best_comp)
        best_ehull_reward_history.append(0.0 if np.isnan(best_ehull) else best_ehull)
        # store weighted rDOS for the best-composite compound (for component visualization)
        best_weighted_rdos_history.append(0.0 if np.isnan(best_rdos) else (gamma * best_rdos))
        # store monotonic max rDOS seen so far (unweighted)
        best_rdos_max_history.append(current_max_rdos)

    # Plot with tighter publication-friendly size and higher DPI
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.plot(best_composite_history, lw=2, color='#1f77b4', label='Best Composite')
    # Best ehull_reward labeled as r_{E_{Hull}} with E italic and Hull roman
    ax.plot(best_ehull_reward_history, lw=1.5, color='#ff7f0e', label=r"Best $r_{E_{\mathrm{Hull}}}$")
    # Best weighted rDOS (component of composite) labeled as r_{DOS}
    ax.plot(best_weighted_rdos_history, lw=1.5, color='#2ca02c', label=r"Best $r_{\mathrm{DOS}}$")

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Score')
    # zero baseline removed per publication formatting
    ax.legend(fontsize=7)
    plt.tight_layout()
    out = out_dir / 'composite_convergence.png'
    fig.savefig(out, dpi=600)
    plt.close(fig)


def main():
    script_dir = Path(__file__).parent
    repo_root = script_dir.parents[2]
    out_dir = script_dir

    # Ensure a subsidiary figures directory exists; we'll move all generated PNGs here
    figures_dir = out_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: compute composite scores and save CSVs
    all_compounds_csv = out_dir / 'all_compounds.csv'
    if not all_compounds_csv.exists():
        print('Error: all_compounds.csv not found in analysis directory')
        sys.exit(1)

    df = pd.read_csv(all_compounds_csv)
    df_sorted = compute_composite(df)
    df_sorted.to_csv(out_dir / 'all_compounds_by_composite_score.csv', index=False)
    df_sorted.head(10).to_csv(out_dir / 'top10_compounds_by_composite_score.csv', index=False)
    print('Saved composite CSVs')

    # Generate figures
    plot_top10(df_sorted, out_dir)
    plot_ehull_vs_rdos(repo_root, out_dir)
    plot_composite_convergence(out_dir)

    # Generate radial tree via existing script (it writes PNG)
    subprocess.run([sys.executable, str(out_dir / 'create_composite_radial_tree.py')], check=False)

    # Move any PNGs generated into the subsidiary figures folder
    for p in out_dir.glob('*.png'):
        try:
            target = figures_dir / p.name
            # Overwrite if exists
            if target.exists():
                target.unlink()
            p.rename(target)
        except Exception:
            pass

    print('\nAll figures written to:', figures_dir)


if __name__ == '__main__':
    main()
