#!/usr/bin/env python3
"""
Create radial tree visualization colored by composite reward.

Composite = beta*ehull_reward(e_hull) + gamma*r_DOS, beta=1.0, gamma=2.5

Blue-red color scheme: blue = higher composite (better), red = lower composite (worse).
"""

import pickle
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import networkx as nx
import pandas as pd
from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mcts_crystal.node import ehull_reward


# Element categories
F_BLOCK = {
    'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
    'Tm', 'Yb', 'Lu', 'Th', 'Pa', 'U', 'Np', 'Pu'
}
TRANSITION_METALS = {
    'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'
}
GROUP_IV = {'Si', 'Ge', 'Sn', 'Pb'}


def reorder_formula_unicode(formula):
    """Reorder formula to RE TM₆ GIV₆ with unicode subscripts."""
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)

    re_elem, tm_elem, giv_elem = None, None, None
    for elem, count in matches:
        if not elem:
            continue
        if elem in F_BLOCK:
            re_elem = elem
        elif elem in TRANSITION_METALS:
            tm_elem = elem
        elif elem in GROUP_IV:
            giv_elem = elem

    if re_elem and tm_elem and giv_elem:
        return f"{re_elem}{tm_elem}₆{giv_elem}₆"
    return formula


def compute_composite(e_hull, r_dos, beta=1.0, gamma=2.5):
    """Compute composite reward (E_form is not part of the reward)."""
    return beta * ehull_reward(e_hull) + gamma * r_dos


def count_descendants(G, node):
    """Count descendants including self using NetworkX reachability to avoid recursion/cycles."""
    try:
        desc = nx.descendants(G, node)
        return 1 + len(desc)
    except Exception:
        return 1


def radial_layout(G, root, radius_step=3.0):
    """Compute a non-recursive radial layout positions based on BFS depth.

    Places nodes at radius = depth * radius_step and evenly spaces nodes
    at the same depth around the circle. This avoids recursion and handles
    cycles by using shortest-path depths from the root.
    """
    pos = {}
    if root is None or root not in G:
        return pos
    pos[root] = (0.0, 0.0)

    try:
        depths = nx.single_source_shortest_path_length(G, root)
    except Exception:
        # Fallback: place all nodes in a circle if shortest paths fail
        nodes = list(G.nodes())
        n = len(nodes)
        for i, nkey in enumerate(nodes):
            theta = 2.0 * math.pi * (i / max(1, n))
            pos[nkey] = (radius_step * math.cos(theta), radius_step * math.sin(theta))
        return pos

    # Group nodes by depth
    depth_groups = {}
    for nkey, d in depths.items():
        depth_groups.setdefault(d, []).append(nkey)

    max_depth = max(depth_groups.keys()) if depth_groups else 0

    for d in range(1, max_depth + 1):
        nodes_at_depth = depth_groups.get(d, [])
        m = len(nodes_at_depth)
        for i, nkey in enumerate(nodes_at_depth):
            theta = 2.0 * math.pi * (i / max(1, m))
            r = d * radius_step
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            pos[nkey] = (x, y)

    return pos


def build_tree_data(mcts):
    """Build tree data from MCTS, computing composite for each node."""
    tree_data = {}

    # Load doscar rewards for r_DOS lookup
    doscar_dict = {}
    if hasattr(mcts, 'stat_dict'):
        for formula, stats in mcts.stat_dict.items():
            if len(stats) >= 6:
                doscar_dict[formula] = stats[5]  # r_dos at index 5

    def traverse(node, node_id=0, parent_id=None):
        formula = node.get_chemical_formula()
        e_form = node.e_form if hasattr(node, 'e_form') else None
        e_hull = node.e_above_hull if hasattr(node, 'e_above_hull') else None
        r_dos = doscar_dict.get(formula, 0.0)

        composite = None
        if e_hull is not None:
            composite = compute_composite(e_hull, r_dos)

        tree_data[node_id] = {
            "formula": formula,
            "e_form": e_form,
            "e_hull": e_hull,
            "r_dos": r_dos,
            "composite": composite,
            "parent_id": parent_id,
            "visit_count": node.t_of_visit
        }

        for i, child in enumerate(node.children):
            child_id = node_id * 100 + i + 1
            traverse(child, child_id, node_id)

    traverse(mcts.root, node_id=0)
    return tree_data


def main():
    script_dir = Path(__file__).parent

    # Load MCTS pickle
    pkl_path = script_dir / 'mcts_object.pkl'
    if not pkl_path.exists():
        print(f"Error: {pkl_path} not found")
        return 1

    print("Loading MCTS pickle...")
    with open(pkl_path, 'rb') as f:
        mcts = pickle.load(f)

    # Build tree data with composite scores
    print("Building tree data...")
    tree_data = build_tree_data(mcts)
    print(f"  {len(tree_data)} nodes in tree")

    # Build a condensed networkx graph collapsing nodes by unique formula
    # Aggregate composite (max), r_dos (max), and sum visit_counts for unique formulas.
    unique_map = {}
    edges = set()
    root_formula = None

    for nid, info in tree_data.items():
        formula = info.get('formula') or f'UNK_{nid}'
        # normalize formula key as string
        key = str(formula)
        if key not in unique_map:
            unique_map[key] = {
                'formulas': [formula],
                'composites': [],
                'e_hulls': [],
                'r_doss': [],
                'visit_count': 0
            }
        if info.get('composite') is not None:
            unique_map[key]['composites'].append(info['composite'])
        if info.get('e_hull') is not None:
            unique_map[key]['e_hulls'].append(info['e_hull'])
        unique_map[key]['r_doss'].append(info.get('r_dos', 0.0))
        unique_map[key]['visit_count'] += int(info.get('visit_count', 0) or 0)

        parent = info.get('parent_id')
        if parent is None:
            root_formula = key
        else:
            parent_formula = tree_data.get(parent, {}).get('formula')
            if parent_formula is None:
                parent_key = f'UNK_{parent}'
            else:
                parent_key = str(parent_formula)
            if parent_key != key:
                edges.add((parent_key, key))

    # Decide which unique nodes to keep (trim low-visit nodes for clarity)
    all_nodes = list(unique_map.items())
    # Keep nodes with at least 2 visits by default
    nodes_keep = [k for k, v in unique_map.items() if v['visit_count'] >= 2]
    # If too many nodes remain, keep top 60 by visit_count
    if len(nodes_keep) > 60:
        sorted_nodes = sorted(unique_map.items(), key=lambda kv: kv[1]['visit_count'], reverse=True)
        nodes_keep = [k for k, _ in sorted_nodes[:60]]
    # If too few nodes kept (e.g., most have 1 visit), instead keep top 40 by visit_count
    if len(nodes_keep) < 10:
        sorted_nodes = sorted(unique_map.items(), key=lambda kv: kv[1]['visit_count'], reverse=True)
        nodes_keep = [k for k, _ in sorted_nodes[:40]]

    nodes_keep_set = set(nodes_keep)

    # Build networkx graph from filtered unique_map
    G = nx.DiGraph()
    for key in nodes_keep:
        info = unique_map[key]
        comp_vals = [v for v in info['composites'] if v is not None]
        comp_agg = max(comp_vals) if comp_vals else None
        rdos_vals = [float(v) for v in info['r_doss'] if v is not None]
        rdos_agg = max(rdos_vals) if rdos_vals else 0.0
        eh_vals = [v for v in info['e_hulls'] if v is not None]
        eh_agg = eh_vals[0] if eh_vals else None
        G.add_node(key, formula=key, composite=comp_agg, e_hull=eh_agg, r_dos=rdos_agg, visit_count=info['visit_count'])

    for a, b in edges:
        if a in nodes_keep_set and b in nodes_keep_set:
            if a not in G:
                G.add_node(a, formula=a, composite=None, e_hull=None, r_dos=0.0, visit_count=0)
            if b not in G:
                G.add_node(b, formula=b, composite=None, e_hull=None, r_dos=0.0, visit_count=0)
            G.add_edge(a, b)

    if root_formula is None or root_formula not in G:
        # fallback to arbitrary kept node
        root_formula = next(iter(G.nodes())) if len(G.nodes()) else None

    print(f"  {len(G.nodes())} nodes after trimming (from {len(unique_map)} unique)")

    # Compute layout and flip 180 degrees
    pos = radial_layout(G, root_formula, radius_step=3.0) if root_formula is not None else {}
    pos = {nid: (-x, -y) for nid, (x, y) in pos.items()} if pos else {}

    # Extract metrics from graph node attributes for coloring
    composites = {n: G.nodes[n].get('composite', None) for n in G.nodes()}
    ehull_rewards = {}
    r_doss = {}
    visit_counts = {}
    for n in G.nodes():
        eh = G.nodes[n].get('e_hull', None)
        try:
            ehull_rewards[n] = ehull_reward(eh) if eh is not None else np.nan
        except Exception:
            ehull_rewards[n] = np.nan
        r_doss[n] = float(G.nodes[n].get('r_dos', 0.0) or 0.0)
        visit_counts[n] = float(G.nodes[n].get('visit_count', 0) or 0.0)

    # Choose readable sequential colormaps
    cmap_comp = cm.get_cmap('viridis')
    arr_comp = [v for v in composites.values() if v is not None and not pd.isna(v)]
    norm_comp = mcolors.Normalize(vmin=min(arr_comp), vmax=max(arr_comp)) if arr_comp else None

    cmap_ehull = cm.get_cmap('Oranges')
    arr_eh = [v for v in ehull_rewards.values() if v is not None and not pd.isna(v)]
    norm_ehull = mcolors.Normalize(vmin=min(arr_eh), vmax=max(arr_eh)) if arr_eh else None

    cmap_rdos = cm.get_cmap('Greens')
    arr_r = [v for v in r_doss.values() if v is not None and not pd.isna(v)]
    norm_rdos = mcolors.Normalize(vmin=min(arr_r), vmax=max(arr_r)) if arr_r else None

    if norm_comp is None:
        print("No composite scores available")
        return 1

    def node_colors_for_from_graph(attr_name, cmap, norm, scale=1.0):
        cols = []
        for nid in G.nodes():
            val = G.nodes[nid].get(attr_name, None)
            if val is None or (isinstance(val, float) and pd.isna(val)) or norm is None:
                cols.append('lightgray')
            else:
                cols.append(cmap(norm(float(val) * scale)))
        return cols

    node_colors_comp = node_colors_for_from_graph('composite', cmap_comp, norm_comp)
    node_colors_ehull = node_colors_for_from_graph('e_hull', cmap_ehull, norm_ehull)
    gamma_vis = 2.5
    node_colors_rdos = node_colors_for_from_graph('r_dos', cmap_rdos, norm_rdos, scale=gamma_vis)

    # Compute per-node entropy from visit counts to separate points visually
    total_visits = sum(visit_counts.values()) if visit_counts else 0.0
    entropy_raw = {}
    for nid, vc in visit_counts.items():
        if total_visits > 0 and vc > 0:
            p = vc / total_visits
            entropy_raw[nid] = -p * math.log(p)
        else:
            entropy_raw[nid] = 0.0
    ent_vals = list(entropy_raw.values()) if entropy_raw else [0.0]
    ent_min, ent_max = min(ent_vals), max(ent_vals)
    ent_norm = {n: (entropy_raw[n] - ent_min) / (ent_max - ent_min) if ent_max > ent_min else 0.0 for n in entropy_raw}

    # Apply an entropy-driven radial offset to positions to spread nodes with different entropy
    extra_radius = 2.0
    pos_entropy = {}
    for nid, (x, y) in pos.items():
        r = math.hypot(x, y)
        theta = math.atan2(y, x)
        ent = ent_norm.get(nid, 0.0)
        r_new = r + ent * extra_radius
        pos_entropy[nid] = (r_new * math.cos(theta), r_new * math.sin(theta))
    pos = pos_entropy

    # Set global font size to 10pt for consistent publication text
    plt.rcParams.update({'font.size': 10})

    # Create a wide figure: 6in wide x 2.75in tall with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(6, 2.75), constrained_layout=True)

    panels = [
        (axes[0], node_colors_comp, cmap_comp, norm_comp, 'Composite Score'),
        (axes[1], node_colors_ehull, cmap_ehull, norm_ehull, r"$r_{E_{\mathrm{Hull}}}$"),
        (axes[2], node_colors_rdos, cmap_rdos, norm_rdos, r"$r_{\mathrm{DOS}}$")
    ]

    labels_abc = ['(a)', '(b)', '(c)']
    for i, (ax, ncols, cmap_m, norm_m, label) in enumerate(panels):
        nx.draw(G, pos, ax=ax, with_labels=False,
                node_color=ncols, edge_color='gray', node_size=120,
                arrows=True, connectionstyle='arc3,rad=0.1',
                edgecolors='black', linewidths=0.6)
        # add panel letter in top-left
        try:
            abc = labels_abc[i]
        except Exception:
            abc = ''
        ax.text(0.02, 0.98, abc, transform=ax.transAxes, va='top', ha='left', fontsize=10, weight='bold')
        ax.axis('equal')
        # colorbar for each panel; place metric label below the colorbar
        sm = cm.ScalarMappable(norm=norm_m, cmap=cmap_m)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.08, pad=0.08, aspect=20)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label(label, fontsize=10)
        try:
            cbar.ax.xaxis.set_label_position('bottom')
            cbar.ax.xaxis.tick_bottom()
        except Exception:
            pass

    # Ensure figures directory exists and save into it
    figures_dir = script_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / 'radial_tree_composite.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    return 0


if __name__ == '__main__':
    exit(main())
