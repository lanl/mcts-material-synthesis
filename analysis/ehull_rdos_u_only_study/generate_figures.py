#!/usr/bin/env python3
"""
Generate publication figures as PNGs directly from MCTS outputs and high-throughput data.

This script replaces the prior gnuplot/prepare pipeline and writes PNG files:
- ehull_vs_rdos.png
- composite_convergence.png
- convergence_by_starting_material.png (reuses sensitivity_studies' starting_material_sweep data)
- radial_tree_composite.png (delegates to create_composite_radial_tree.py)

It expects to be run from `analysis/ehull_rdos_u_only_study/` where the MCTS
run outputs (`all_compounds.csv`, `convergence_history.csv`, `mcts_object.pkl`) are
present (the example `run_study.sh` places them there). It also expects the
high-throughput cache `high_throughput_mace_results.full.csv` and
`doscar_peaks_data_with_U.csv` to be available at the repository root for the
`ehull_vs_rdos` figure if the analysis CSV is not present (rDOS is always
computed in real time from the raw peaks file - no precomputed rewards cache).
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess
import sys
import shutil
import re
import json

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthesized_compounds import SYNTHESIZED_COMPOUNDS
from create_composite_radial_tree import F_BLOCK, TRANSITION_METALS, GROUP_IV

DEFAULT_GAMMA = 0.0001


def load_gamma(default: float = DEFAULT_GAMMA) -> float:
    """Load the single gamma (rDOS weight) from config.json, if present.

    gamma is used both as the composite-score weight (beta*ehull_reward + gamma*r_DOS)
    and to scale r_DOS for plot overlays - there is only one gamma value, not a
    separate "gamma_prefactor" for plots.
    """
    config_path = Path(__file__).resolve().parents[2] / 'config.json'
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            rollout_method = config.get('rollout_method')
            if rollout_method is not None and not str(rollout_method).startswith('ehull_rdos'):
                print(f"Warning: config.json rollout_method={rollout_method!r}, "
                      "but figures assume the ehull+rDOS composite score (beta=1.0, gamma from config).")
            return float(config.get('gamma', default))
        except Exception:
            pass
    return default


def compute_composite(df, beta=1.0, gamma=None):
    if gamma is None:
        gamma = load_gamma()
    try:
        from mcts_crystal.node import ehull_reward
    except Exception:
        # Fallback: define local ehull_reward matching mcts_crystal.node implementation
        def ehull_reward(e_hull: float) -> float:
            import numpy as _np
            return -_np.tanh(120.0 * (e_hull - 0.05))
    df = df.copy()
    df['name'] = df.get('formula', df.get('name'))
    # ensure r_DOS is computed from canonical doscar rewards if not present
    # Do NOT store gamma-weighted values in r_DOS; r_DOS should be the raw [0,1] reward.
    if 'r_DOS' not in df.columns or df['r_DOS'].isnull().all() or (df['r_DOS'].astype(float) == 0.0).all():
        # load master mapping from repo root
        repo_root = Path(__file__).parents[2]
        df_master, dos_by_key = load_master_mace(repo_root)

        # element-set key function (ignore f-block differences)
        F_BLOCK = {'Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Th','Pa','U','Np','Pu','Ac'}
        def parse_elems(s):
            if pd.isna(s):
                return []
            if '-' in str(s):
                parts = [p for p in re.split('[^A-Za-z]', str(s)) if p]
                return parts
            return re.findall(r'[A-Z][a-z]?', str(s))

        def key_from_name(name):
            elems = parse_elems(name)
            return tuple(sorted([e for e in elems if e not in F_BLOCK]))

        # default to 0.0
        df['r_DOS'] = 0.0
        for idx, row in df.iterrows():
            key = key_from_name(row.get('name', row.get('formula', '')))
            if key in dos_by_key:
                df.at[idx, 'r_DOS'] = float(dos_by_key[key])
            else:
                # fallback: try normalized name match against master if available
                if df_master is not None:
                    nm = str(row.get('name', row.get('formula', '')))
                    # try exact match on name/formula
                    try:
                        match = df_master[(df_master.get('name', df_master.get('formula')) == nm)].iloc[0]
                        # if master has r_dos or similar, use it
                        if 'r_dos' in df_master.columns and not pd.isna(match.get('r_dos')):
                            df.at[idx, 'r_DOS'] = float(match.get('r_dos'))
                    except Exception:
                        pass
    df['ehull_reward'] = df['e_above_hull'].apply(ehull_reward)
    # Keep raw r_DOS as canonical; composite uses gamma weighting when computing score
    df['weighted_r_DOS'] = gamma * df['r_DOS']
    df['composite_score'] = beta * df['ehull_reward'] + df['weighted_r_DOS']
    df_sorted = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    return df_sorted


def load_master_mace(repo_root: Path):
    """Load canonical high-throughput MACE CSV and DOS rewards (computed in real
    time from raw peak data - there is no precomputed rewards cache).
    Returns (df_mace, dos_by_key) where dos_by_key maps element-set keys (the
    TM/group-IV pair, f-block element stripped) to a DOS reward.

    doscar_peaks_data_with_U.csv has a separate row - and a genuinely
    different reward - per f-block element (e.g. 'U-Ge-Fe' vs 'Er-Ge-Fe' are
    different compounds with different DOS peaks), so for each (TM, group-IV)
    pair we prefer the exact uranium-row reward where one exists. We only
    fall back to the max across other f-block substitutions (a same-framework
    proxy) when uranium's own row is missing - never as a default, since that
    would substitute a different compound's reward for uranium's.
    """
    # Prefer the file in this repo; only search parent/sibling folders as a fallback
    # (this repo may sit next to other mcts_materials checkouts with stale copies).
    MAX_PARENT_DEPTH = 4
    search_roots = [repo_root] + list(repo_root.parents)[:MAX_PARENT_DEPTH]

    def find_canonical(filename):
        local = repo_root / filename
        if local.exists():
            return local
        for r in search_roots:
            try:
                candidates = list(Path(r).rglob(filename))
            except OSError:
                continue
            if candidates:
                return candidates[0]
        return repo_root / filename

    mace_csv = find_canonical('high_throughput_mace_results.full.csv')
    df_mace = None
    dos_by_key = {}
    if mace_csv.exists():
        try:
            df_mace = pd.read_csv(mace_csv)
        except Exception:
            df_mace = None

    # Compute DOS rewards in real time from the raw peaks file (prefer local repo copy)
    from mcts_crystal.doscar_utils import DoscarRewardLookup
    peaks_csv = find_canonical('doscar_peaks_data_with_U.csv')
    dos_dict = DoscarRewardLookup(peaks_file=str(peaks_csv)).rewards_dict

    F_BLOCK = {'Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Th','Pa','U','Np','Pu','Ac'}

    def parse_elems(s):
        if pd.isna(s):
            return []
        if '-' in str(s):
            parts = [p for p in re.split('[^A-Za-z]', str(s)) if p]
            return parts
        return re.findall(r'[A-Z][a-z]?', str(s))

    # collapse compound_name -> reward into element-set keys (TM/group-IV pair,
    # f-block element stripped). dos_by_key_fallback is the max across whatever
    # f-block substitutions exist for that pair; dos_by_key_uranium is the
    # exact 'U-...' row only. Uranium's own data wins wherever it exists.
    dos_by_key_fallback = {}
    dos_by_key_uranium = {}
    for name, val in dos_dict.items():
        try:
            v = float(val)
        except Exception:
            continue
        elems = parse_elems(name)
        key = tuple(sorted([e for e in elems if e not in F_BLOCK]))
        if not key:
            continue
        dos_by_key_fallback[key] = max(dos_by_key_fallback.get(key, float('-inf')), v)
        if 'U' in elems:
            dos_by_key_uranium[key] = v

    dos_by_key = dos_by_key_fallback
    dos_by_key.update(dos_by_key_uranium)

    return df_mace, dos_by_key


def plot_ehull_vs_rdos(repo_root: Path, out_dir: Path):
    comp_csv = out_dir / 'all_compounds_by_composite_score.csv'
    df_u = None
    # x-axis for this plot is gamma * r_DOS (the actual composite-score component),
    # applied consistently to the backdrop and every overlay below.
    gamma = load_gamma()

    # Delegates to load_master_mace() for both the high-throughput CSV and the
    # DOS-reward lookup, so this backdrop always uses the same (uranium-row-
    # preferring, not cross-lanthanide-max) r_DOS as compute_composite() and
    # the top15 table - see load_master_mace()'s docstring for why that matters.
    df_mace, dos_by_key = load_master_mace(repo_root)

    def _make_key_from_formula(formula):
        if pd.isna(formula):
            return ()
        s = str(formula)
        if '-' in s:
            parts = [p for p in re.split('[^A-Za-z]', s) if p]
        else:
            parts = re.findall(r'[A-Z][a-z]?', s)
        F_BLOCK = {'Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Th','Pa','U','Np','Pu','Ac'}
        return tuple(sorted([e for e in parts if e not in F_BLOCK]))

    if df_mace is not None:
        # attach r_dos by element-set key (TM/group-IV pair, f-block element stripped)
        df_mace['name'] = df_mace.get('name', df_mace.get('formula'))
        df_mace['r_dos'] = 0.0
        for idx, row in df_mace.iterrows():
            key = _make_key_from_formula(row['name'])
            if key in dos_by_key:
                df_mace.at[idx, 'r_dos'] = dos_by_key[key]
        if 'name' not in df_mace.columns and 'formula' in df_mace.columns:
            df_mace['name'] = df_mace['formula']

        # Filter to U-containing compounds (exclude Ce)
        def elem_set(name):
            if pd.isna(name):
                return set()
            s = str(name)
            if '-' in s:
                parts = [p for p in re.split('[^A-Za-z]', s) if p]
                return set([p.capitalize() for p in parts])
            parts = re.findall(r'[A-Z][a-z]?', s)
            return set(parts)

        df_mace['elem_set'] = df_mace['name'].apply(elem_set)
        df_u = df_mace[df_mace['elem_set'].apply(lambda s: 'U' in s and 'Ce' not in s)].copy()

    # If we still don't have a mace-derived U list, fall back to the analysis CSV (MCTS outputs)
    if df_u is None and comp_csv.exists():
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

    if df_u is None or df_u.empty:
        print('Skipping ehull_vs_rdos: no U-containing data found')
        return

    # Some entries may not map; attempt conversion using same heuristic as DoscarRewardLookup
    # For robustness, keep 0.0 where not found.

    # attempt to load experimental compounds list (if provided elsewhere in repo)
    exp_src = repo_root / 'redo_mcts_materials' / 'experimental_comparison' / 'compounds_filtered.dat'
    exp_dst = out_dir / 'compounds_filtered.dat'
    df_exp = None
    if exp_src.exists():
        try:
            df_exp = pd.read_csv(exp_src, sep=r"\s+", comment='#', header=None,
                                names=['name', 'e_form', 'e_hull'])
            # copy into analysis folder for consistency and ignore in git
            if not exp_dst.exists():
                shutil.copy(exp_src, exp_dst)
        except Exception:
            df_exp = None

    fig, ax = plt.subplots(figsize=(4, 4))
    # Use r_dos (already present) or fall back to r_DOS column; plot gamma * r_DOS
    rdos_vals = df_u.get('r_dos', df_u.get('r_DOS', pd.Series(0.0))).astype(float) * gamma
    ax.scatter(rdos_vals, df_u['e_above_hull'].astype(float), s=6, color='#D0D0D0', label='All Compounds (U-containing)')

    # Normalize helper
    def _norm(s):
        if pd.isna(s):
            return ''
        return ''.join([c for c in str(s).lower() if c.isalnum()])

    # prepare df_u name_norm
    if 'name' in df_u.columns:
        df_u['name_norm'] = df_u['name'].apply(_norm)
    else:
        df_u['name_norm'] = df_u.get('formula', '').apply(_norm)

    # synthesized names (successful) - see synthesized_compounds.py
    synthesized_names = SYNTHESIZED_COMPOUNDS
    def elem_set(name):
        if pd.isna(name):
            return set()
        s = str(name)
        if '-' in s:
            parts = [p for p in re.split('[^A-Za-z]', s) if p]
            return set([p.capitalize() for p in parts])
        parts = re.findall(r'[A-Z][a-z]?', s)
        return set(parts)

    synth_sets = [elem_set(n) for n in synthesized_names]

    # experimental attempted compounds (from df_exp) and classify as successful/unsuccessful
    exp_unsucc_x, exp_unsucc_y = [], []
    exp_succ_x, exp_succ_y = [], []
    if df_exp is not None and not df_exp.empty:
        df_exp['name_norm'] = df_exp['name'].apply(_norm)
        name_map = {n: r for n, r in zip(df_u['name_norm'], df_u.to_dict('records'))}
        for _, r in df_exp.iterrows():
            nrm = r['name_norm']
            if nrm in name_map:
                row = name_map[nrm]
                x = float(row.get('r_dos', row.get('r_DOS', 0.0))) * gamma
                y = float(row.get('e_above_hull', r['e_hull']))
                # classify by element set equality
                if any(elem_set(r['name']) == s for s in synth_sets):
                    exp_succ_x.append(x); exp_succ_y.append(y)
                else:
                    exp_unsucc_x.append(x); exp_unsucc_y.append(y)

    # plot experimental attempted: unsuccessful = purple unfilled squares
    if exp_unsucc_x:
        ax.scatter(exp_unsucc_x, exp_unsucc_y, s=80, marker='s', facecolors='none', edgecolors='#9467bd', linewidths=1.2, label='Unsuccessful Synthesis')
    # plot experimental successful: purple filled squares
    if exp_succ_x:
        ax.scatter(exp_succ_x, exp_succ_y, s=100, marker='s', facecolors='#9467bd', edgecolors='#9467bd', linewidths=0.8, label='Successful Synthesis')

    # Top10 overlay if available (look in out_dir) - teal filled triangles for MCTS predictions
    composite_csv = out_dir / 'all_compounds_by_composite_score.csv'
    if composite_csv.exists():
        df_comp = pd.read_csv(composite_csv)
        top10 = df_comp.head(10)
        mace_lookup = dict(zip(df_u['name'], zip(df_u['e_above_hull'], df_u.get('r_dos', df_u.get('r_DOS', pd.Series(0.0))))))
        xs, ys = [], []
        for _, row in top10.iterrows():
            name = row['name'] if 'name' in row else row.get('formula')
            if name in mace_lookup:
                e_hull, rdos = mace_lookup[name]
                xs.append(float(rdos) * gamma)
                ys.append(e_hull)
        if xs:
            ax.scatter(xs, ys, s=80, color='#5BC0EB', marker='^', edgecolors='#5BC0EB', label='Top 10 (MCTS)')

    # (Top10 plotting handled earlier to control marker style)

    ax.set_xlabel(r"$\gamma \cdot r_{\mathrm{DOS}}$" + f" (γ={gamma:g})")
    ax.set_ylabel(r"$E_{\mathrm{Hull}}$ (eV/atom)")
    ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
    # Create a clean legend with specific marker styles so Top10 appears as a triangle
    from matplotlib.lines import Line2D
    legend_handles = []
    # All compounds: small light-gray circle (no line)
    legend_handles.append(Line2D([0], [0], marker='o', linestyle='None', markerfacecolor='#D0D0D0', markeredgecolor='#D0D0D0', markersize=6, label='All Compounds'))
    # Top10 (MCTS): light blue filled triangle (no line)
    legend_handles.append(Line2D([0], [0], marker='^', linestyle='None', markerfacecolor='#5BC0EB', markeredgecolor='#5BC0EB', markersize=8, label='Top 10 (MCTS)'))
    # Unsuccessful: purple unfilled square (no line)
    legend_handles.append(Line2D([0], [0], marker='s', linestyle='None', markerfacecolor='none', markeredgecolor='#9467bd', markersize=8, label='Unsuccessful Synthesis'))
    # Successful: purple filled square with purple edge (no line)
    legend_handles.append(Line2D([0], [0], marker='s', linestyle='None', markerfacecolor='#9467bd', markeredgecolor='#9467bd', markersize=9, label='Successful Synthesis'))
    ax.legend(handles=legend_handles, fontsize=8)
    plt.tight_layout()
    out = out_dir / 'ehull_vs_rdos.png'
    fig.savefig(out, dpi=300)
    plt.close(fig)


def format_name_u_tm_giv(name):
    """Reorder a 1-6-6 formula as U, transition metal, group IV with the 6's
    rendered as LaTeX subscripts, e.g. 'Fe6Ge6U' -> 'UFe$_6$Ge$_6$'.

    Reuses the same element categorization as create_composite_radial_tree.py
    so the ordering convention stays consistent across figures and tables.
    """
    matches = re.findall(r'([A-Z][a-z]?)(\d*)', str(name))
    re_elem, tm_part, giv_part = None, None, None
    for elem, count in matches:
        if not elem:
            continue
        if elem in F_BLOCK:
            re_elem = elem
        elif elem in TRANSITION_METALS:
            tm_part = (elem, count)
        elif elem in GROUP_IV:
            giv_part = (elem, count)
    if re_elem and tm_part and giv_part:
        tm_elem, tm_n = tm_part
        giv_elem, giv_n = giv_part
        tm_sub = f"$_{{{tm_n}}}$" if tm_n else ''
        giv_sub = f"$_{{{giv_n}}}$" if giv_n else ''
        return f"{re_elem}{tm_elem}{tm_sub}{giv_elem}{giv_sub}"
    return str(name)


def _formula_key(name):
    """Element-set key (sorted, non-f-block) used to match a formula across
    datasets regardless of element ordering, e.g. 'Fe6Ge6U' -> ('Fe', 'Ge')."""
    if pd.isna(name):
        return ()
    s = str(name)
    if '-' in s:
        parts = [p for p in re.split('[^A-Za-z]', s) if p]
    else:
        parts = re.findall(r'[A-Z][a-z]?', s)
    return tuple(sorted(p for p in parts if p not in F_BLOCK))


def compute_global_u_only_ranks(repo_root: Path):
    """Rank every compound in the full high-throughput U-only design space
    (U present, no other f-block element) by the same beta=1.0/gamma composite
    score used everywhere else, so MCTS top-15 results can be checked against
    the true best-in-space compounds (search coverage, not just local quality).

    Returns {element_key: global_rank} with rank 1 = best composite score.
    """
    df_mace, dos_by_key = load_master_mace(repo_root)
    if df_mace is None or not len(df_mace):
        return {}

    try:
        from mcts_crystal.node import ehull_reward
    except Exception:
        def ehull_reward(x):
            return -np.tanh(120.0 * (x - 0.05))

    gamma = load_gamma()
    df = df_mace.copy()
    df['name'] = df.get('name', df.get('formula'))
    df['elem_set'] = df['name'].apply(lambda n: set(re.findall(r'[A-Z][a-z]?', str(n))))
    # U-only design space: contains U, and no other f-block/rare-earth element
    df = df[df['elem_set'].apply(lambda s: 'U' in s and not (s & (F_BLOCK - {'U'})))].copy()
    df['key'] = df['name'].apply(_formula_key)
    df['r_dos'] = df['key'].map(lambda k: dos_by_key.get(k, 0.0))
    df['composite_score'] = df['e_above_hull'].apply(ehull_reward) + gamma * df['r_dos']
    df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    return {key: rank for rank, key in enumerate(df['key'], start=1)}


def write_top15_table(df_sorted: pd.DataFrame, out_dir: Path, repo_root: Path):
    """Write a LaTeX table of the top 15 compounds with requested columns."""
    tables_dir = out_dir / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    top15 = df_sorted.head(15).copy()
    gamma = load_gamma()
    global_ranks = compute_global_u_only_ranks(repo_root)
    # ensure columns exist
    if 'r_DOS' not in top15.columns and 'dos_reward' in top15.columns:
        top15['r_DOS'] = top15['dos_reward']
    if 'weighted_r_DOS' not in top15.columns:
        top15['weighted_r_DOS'] = top15['r_DOS'] * gamma
    if 'ehull_reward' not in top15.columns and 'e_above_hull' in top15.columns:
        # recompute if possible
        try:
            from mcts_crystal.node import ehull_reward
        except Exception:
            def ehull_reward(x):
                import numpy as _np
                return -_np.tanh(120.0 * (x - 0.05))
        top15['ehull_reward'] = top15['e_above_hull'].apply(ehull_reward)

    # synthesized detection by element sets - see synthesized_compounds.py
    synthesized_names = SYNTHESIZED_COMPOUNDS
    import re
    def elem_set(name):
        if pd.isna(name):
            return set()
        s = str(name)
        if '-' in s:
            parts = [p for p in re.split('[^A-Za-z]', s) if p]
            return set([p.capitalize() for p in parts])
        parts = re.findall(r'[A-Z][a-z]?', s)
        return set(parts)

    synth_sets = [elem_set(n) for n in synthesized_names]

    rows = []
    for rank, (_, r) in enumerate(top15.iterrows(), start=1):
        name = r.get('name', r.get('formula', ''))
        rdos = float(r.get('weighted_r_DOS', 0.0)) if pd.notna(r.get('weighted_r_DOS', None)) else 0.0
        ehull_r = float(r.get('ehull_reward', 0.0)) if pd.notna(r.get('ehull_reward', None)) else 0.0
        comp = float(r.get('composite_score', 0.0)) if pd.notna(r.get('composite_score', None)) else 0.0
        ehull = float(r.get('e_above_hull', np.nan)) if pd.notna(r.get('e_above_hull', None)) else np.nan
        is_synth = any(elem_set(name) == s for s in synth_sets)
        display_name = format_name_u_tm_giv(name)
        global_rank = global_ranks.get(_formula_key(name))
        rows.append((rank, global_rank, display_name, rdos, ehull_r, comp, ehull, 'Yes' if is_synth else 'No'))

    eol = ' \\\\\n'  # LaTeX row terminator (literal "\\") followed by a real newline
    tex_path = tables_dir / 'top15_u_only.tex'
    with open(tex_path, 'w') as f:
        f.write(f'% Top 15 compounds (U-only study). gamma={gamma:g}. True Rank = rank within the\n')
        f.write(f'% full {len(global_ranks)}-compound exhaustive U-only design space (search coverage check).\n')
        f.write('\\begin{tabular}{rrlrrrrc}\n')
        f.write('\\toprule\n')
        f.write('Rank & True Rank & Name & $\\gamma \\cdot r_{\\mathrm{DOS}}$ ($\\gamma$='
                f'{gamma:g}) & $r_{{E_{{\\mathrm{{Hull}}}}}}$ & Composite & E\\_hull & Synth' + eol)
        f.write('\\midrule\n')
        for rank, global_rank, name, rdos, ehull_r, comp, ehull, synth in rows:
            gr = str(global_rank) if global_rank is not None else '--'
            f.write(f"{rank} & {gr} & {name} & {rdos:.4f} & {ehull_r:.4f} & {comp:.4f} & {ehull:.4f} & {synth}" + eol)
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')

    print('Wrote LaTeX table:', tex_path)


def _set_plateau_xlim(ax, curves, buffer_frac=0.25, tol_frac=0.01, ignore_first=1):
    """Truncate the x-axis around the transient of one or more same-length
    curves: find the last index at which *any* curve still differs from its
    own final value by more than tol_frac of its settled range, then size the
    axis so that point sits at (1 - buffer_frac) of the width, leaving the
    rest as flat-plateau context.

    A relative tol_frac (rather than exact equality) matters once curves are
    averaged across seeds: a single seed nudging the mean by a negligible,
    sub-percent amount very late in the run would otherwise count as "still
    changing" and drag the whole x-axis back out to the full run length.

    ignore_first excludes a leading pre-search sentinel (e.g. best_reward =
    -10.0 at iteration 0) from both the range and "still changing" checks, so
    it doesn't swamp the real post-search dynamic range.

    Using the slowest-to-settle curve (max plateau-start across `curves`)
    keeps every curve's full transient visible when comparing several at once.
    """
    plateau_starts = []
    n_iter = None
    for arr in curves:
        arr = np.asarray(arr, dtype=float)
        if arr.size <= ignore_first:
            continue
        n_iter = len(arr) if n_iter is None else max(n_iter, len(arr))
        body = arr[ignore_first:]
        final_val = body[-1]
        rng = np.nanmax(body) - np.nanmin(body)
        tol = tol_frac * rng if rng > 0 else 1e-9 * max(1.0, abs(final_val))
        changed = np.where(np.abs(body - final_val) > tol)[0]
        plateau_starts.append((changed[-1] + 1 + ignore_first) if changed.size > 0 else ignore_first)
    if not plateau_starts or n_iter is None:
        return
    plateau_start = min(max(plateau_starts), n_iter - 1)
    if plateau_start > 0:
        xmax = min(plateau_start / (1.0 - buffer_frac), n_iter - 1)
        ax.set_xlim(0, xmax)


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
    best_ehull_reward_history = []
    # Track raw best rDOS history (plot raw r_DOS, not gamma-weighted)
    best_weighted_rdos_history = []
    best_rdos_max_history = []

    gamma = load_gamma()

    # best_reward is MCTS's own exact running-max composite score (tracked
    # directly in MCTS.max_reward_history) - no need to reconstruct an
    # approximation of it from the best_e_form/best_e_hull/best_rdos formula
    # columns below, which can miss compounds that were visited but never
    # individually flagged as the best e_form/e_hull/rDOS.
    best_composite_history = df_conv['best_reward'].tolist() if 'best_reward' in df_conv.columns else []

    for _, row in df_conv.iterrows():
        for col in ['best_e_form_formula', 'best_e_hull_formula', 'best_rdos_formula']:
            if col in df_conv.columns and pd.notna(row[col]):
                formulas_seen.add(row[col])

        # Determine the component breakdown (ehull_reward, r_DOS) of whichever
        # of the formulas above currently has the highest composite score -
        # still an approximation, since the column above doesn't carry its
        # own component breakdown, but it's the best proxy available without
        # re-deriving the actual winning node's components from the pickle.
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

        if current_max_rdos == float('-inf'):
            current_max_rdos = 0.0

        best_ehull_reward_history.append(0.0 if np.isnan(best_ehull) else best_ehull)
        # store gamma * r_DOS for the best-composite compound (the actual composite-score component)
        best_weighted_rdos_history.append(0.0 if np.isnan(best_rdos) else (best_rdos * gamma))
        # store monotonic max rDOS seen so far (unweighted)
        best_rdos_max_history.append(current_max_rdos)

    if not best_composite_history:
        # Fallback for older convergence_history.csv files without best_reward
        best_composite_history = [0.0] * len(df_conv)

    # Iteration 0 is a pre-search sentinel (best_reward = -10.0, not a real
    # composite score), which would otherwise dominate the y-axis and
    # compress the rest of the curve into an unreadable sliver.
    start = 1 if len(best_composite_history) > 1 else 0

    # Plot with tighter publication-friendly size and higher DPI
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.plot(range(start, len(best_composite_history)), best_composite_history[start:],
            lw=2, color='#1f77b4', label='Best Composite')
    # Best ehull_reward labeled as r_{E_{Hull}} with E italic and Hull roman
    ax.plot(range(start, len(best_ehull_reward_history)), best_ehull_reward_history[start:],
            lw=1.5, color='#ff7f0e', label=r"Best $r_{E_{\mathrm{Hull}}}$")
    # Best gamma * r_DOS (component of composite), explicitly labeled with the gamma factor
    ax.plot(range(start, len(best_weighted_rdos_history)), best_weighted_rdos_history[start:],
            lw=1.5, color='#2ca02c', label=r"Best $\gamma \cdot r_{\mathrm{DOS}}$")

    _set_plateau_xlim(ax, [best_composite_history])

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Score')
    # zero baseline removed per publication formatting
    ax.legend(fontsize=7)
    plt.tight_layout()
    out = out_dir / 'composite_convergence.png'
    fig.savefig(out, dpi=600)
    plt.close(fig)


def plot_convergence_by_starting_material(mcts_materials_root: Path, out_dir: Path):
    """Compare convergence dynamics across different starting materials.

    Reuses sensitivity_studies/results/starting_material_sweep/convergence_data.csv
    instead of re-running MCTS: that sweep already uses this study's exact
    composite reward (rollout_method='ehull_rdos', beta=1.0, gamma=0.0001 -
    see sensitivity_studies/scripts/common.py BASELINE), varying only
    transition_metal/group_iv, with 5 seeds per starting material. Each line
    below is the mean best-composite-so-far across those 5 seeds, with a
    shaded band showing seed-to-seed spread.
    """
    sweep_csv = (mcts_materials_root / 'sensitivity_studies' / 'results'
                 / 'starting_material_sweep' / 'convergence_data.csv')
    if not sweep_csv.exists():
        print('Skipping convergence_by_starting_material: missing', sweep_csv)
        return

    df = pd.read_csv(sweep_csv)
    value_order = list(df['value'].unique())  # preserve the sweep's own (graph-distance) ordering
    stats = df.groupby(['value', 'iteration'])['best_reward'].agg(['mean', 'std']).reset_index()

    fig, ax = plt.subplots(figsize=(4, 3.2))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    curves_for_xlim = []
    for color, value in zip(colors, value_order):
        g = stats[stats['value'] == value].sort_values('iteration').reset_index(drop=True)
        # Keep iteration 0 in the array used for x-axis truncation below, so
        # its index stays aligned with real iteration numbers; the sentinel
        # best_reward=-10.0 at iteration 0 is excluded only from the drawn
        # line/band so it doesn't dominate the y-axis (same as composite_convergence).
        curves_for_xlim.append(g['mean'].to_numpy())

        g_plot = g[g['iteration'] >= 1]
        label = format_name_u_tm_giv(value.split(' (')[0])
        ax.plot(g_plot['iteration'], g_plot['mean'], lw=1.8, color=color, label=label)
        std = g_plot['std'].fillna(0.0)
        ax.fill_between(g_plot['iteration'], g_plot['mean'] - std, g_plot['mean'] + std,
                         color=color, alpha=0.2, linewidth=0)

    _set_plateau_xlim(ax, curves_for_xlim)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best Composite Score')
    ax.legend(fontsize=7, title='Starting material', title_fontsize=7)
    plt.tight_layout()
    out = out_dir / 'convergence_by_starting_material.png'
    fig.savefig(out, dpi=600)
    plt.close(fig)
    print(f'Saved: {out}')


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
    # write LaTeX table of top 15
    try:
        write_top15_table(df_sorted, out_dir, repo_root)
    except Exception:
        pass

    # Generate figures
    plot_ehull_vs_rdos(repo_root, out_dir)
    plot_composite_convergence(out_dir)
    plot_convergence_by_starting_material(script_dir.parents[1], out_dir)

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
