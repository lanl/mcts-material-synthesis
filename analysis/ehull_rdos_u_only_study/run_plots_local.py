#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

out_dir = Path('.')
all_csv = out_dir / 'all_compounds.csv'
conv_csv = out_dir / 'convergence_history.csv'

if not all_csv.exists():
    print('Missing all_compounds.csv')
    sys.exit(1)

def ehull_reward(e_hull):
    return -np.tanh(120.0 * (e_hull - 0.05))

print('Loading', all_csv)
df = pd.read_csv(all_csv)
# ensure columns
if 'formula' in df.columns:
    df['name'] = df['formula']
else:
    df['name'] = df.get('name', df.index.astype(str))

if 'dos_reward' in df.columns:
    df['r_DOS'] = df['dos_reward']
else:
    # compute r_DOS in real time from a nearby doscar_peaks_data_with_U.csv (no
    # precomputed rewards cache), matched via element-set keys
    MAX_PARENT_DEPTH = 4
    search_roots = [out_dir] + list(out_dir.parents)[:MAX_PARENT_DEPTH]
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
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from mcts_crystal.doscar_utils import DoscarRewardLookup
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

if 'e_above_hull' not in df.columns:
    df['e_above_hull'] = df.get('e_hull', 0.0)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mcts_crystal.cli import load_config
_config = load_config(str(Path(__file__).resolve().parents[2] / 'config.json'))
beta = float(_config.get('beta', 1.0))
gamma = float(_config.get('gamma', 0.0001))

print('Computing composite scores')
df['ehull_reward'] = df['e_above_hull'].apply(ehull_reward)
df['weighted_r_DOS'] = gamma * df['r_DOS']
df['composite_score'] = beta * df['ehull_reward'] + df['weighted_r_DOS']
df_sorted = df.sort_values('composite_score', ascending=False).reset_index(drop=True)

df_sorted.to_csv(out_dir / 'all_compounds_by_composite_score.csv', index=False)
df_sorted.head(10).to_csv(out_dir / 'top10_compounds_by_composite_score.csv', index=False)

# top10 horizontal stacked
print('Plotting top10')
top10 = df_sorted.head(10)
fig, ax = plt.subplots(figsize=(4,4))
ranks = np.arange(1, len(top10)+1)
ehull_vals = top10['ehull_reward'].values
rdos_vals = top10['r_DOS'].values * gamma
ax.barh(ranks, ehull_vals, color='#ff7f0e', edgecolor='k')
ax.barh(ranks, rdos_vals, left=ehull_vals, color='#2ca02c', edgecolor='k')
ax.set_yticks(ranks)
ax.set_yticklabels(top10['name'].values)
ax.invert_yaxis()
ax.set_xlabel('Composite components')
plt.tight_layout()
fig.savefig(out_dir / 'top10_by_composite.png', dpi=300)
plt.close(fig)

# grouped bars
fig, ax = plt.subplots(figsize=(4,4))
x = np.arange(len(top10))
width=0.35
ax.bar(x-width/2, top10['e_above_hull'].values, width, label='E_hull', color='#ff7f0e')
ax.bar(x+width/2, gamma * top10['r_DOS'].values, width, label=f'{gamma}*r_DOS', color='#2ca02c')
ax.set_xticks(x)
ax.set_xticklabels(top10['name'].values, rotation=45, ha='right')
ax.set_ylabel('E_hull / rDOS')
ax.legend()
plt.tight_layout()
fig.savefig(out_dir / 'top10_ehull_rdos_bars.png', dpi=300)
plt.close(fig)

# composite convergence
if conv_csv.exists():
    print('Plotting convergence')
    df_conv = pd.read_csv(conv_csv)
    composite_lookup = dict(zip(df_sorted['name'], df_sorted['composite_score']))
    formulas_seen = set()
    best_history = []
    for _, row in df_conv.iterrows():
        for col in ['best_e_form_formula','best_e_hull_formula','best_rdos_formula']:
            if col in df_conv.columns and pd.notna(row[col]):
                formulas_seen.add(row[col])
        best = float('-inf')
        for f in formulas_seen:
            if f in composite_lookup:
                best = max(best, composite_lookup[f])
        if best==float('-inf'):
            best=0.0
        best_history.append(best)
    fig, ax = plt.subplots(figsize=(4,4))
    ax.plot(best_history, lw=2, color='#1f77b4')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Best Composite Score')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    fig.savefig(out_dir / 'composite_convergence.png', dpi=300)
    plt.close(fig)
else:
    print('No convergence_history.csv present; skipping convergence plot')

# ehull vs rdos using repo-root cache
repo_root = Path('..')
if (repo_root / 'high_throughput_mace_results.full.csv').exists() and (repo_root / 'doscar_peaks_data_with_U.csv').exists():
    print('Plotting ehull vs rdos')
    df_mace = pd.read_csv(repo_root / 'high_throughput_mace_results.full.csv')
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from mcts_crystal.doscar_utils import DoscarRewardLookup
    dos_dict = DoscarRewardLookup(peaks_file=str(repo_root / 'doscar_peaks_data_with_U.csv')).rewards_dict
    def has_element(name, elem):
        import re
        pattern = r'(?<![a-z])' + elem + r'(?![a-z])'
        return bool(re.search(pattern, name))
    df_u = df_mace[df_mace['name'].apply(lambda x: has_element(x, 'U'))].copy()
    df_u = df_u[~df_u['name'].apply(lambda x: has_element(x, 'Ce'))].copy()
    df_u['r_dos'] = df_u['name'].map(dos_dict).fillna(0)
    fig, ax = plt.subplots(figsize=(4,4))
    ax.scatter(gamma * df_u['r_dos'], df_u['e_above_hull'], s=6, color='#D0D0D0', label='All Compounds')
    # overlay top10
    xs=[]; ys=[]
    for _,row in top10.iterrows():
        name=row['name']
        matched = df_u[df_u['name']==name]
        if not matched.empty:
            xs.append(gamma*matched.iloc[0]['r_dos'])
            ys.append(matched.iloc[0]['e_above_hull'])
    if xs:
        ax.scatter(xs, ys, s=30, color='#17BECF', label='Top10')
    ax.set_xlabel(f'{gamma} * r_DOS')
    ax.set_ylabel('E_hull (eV/atom)')
    ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
    plt.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / 'ehull_vs_rdos.png', dpi=300)
    plt.close(fig)
else:
    print('Skipping ehull_vs_rdos: missing cache or doscar_peaks_data_with_U.csv')

print('Done')
