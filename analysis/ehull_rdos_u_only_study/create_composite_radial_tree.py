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
    """Recursively count descendants including self."""
    children = list(G.successors(node))
    return 1 + sum(count_descendants(G, c) for c in children)


def radial_layout(G, root, radius_step=3.0):
    """Compute radial layout positions."""
    pos = {}
    pos[root] = (0, 0)

    def layout_children(node, depth, angle_start, angle_end):
        children = list(G.successors(node))
        if not children:
            return

        total_desc = sum(count_descendants(G, c) for c in children)
        if total_desc == 0:
            total_desc = len(children)

        angle_range = angle_end - angle_start
        current_angle = angle_start

        for child in children:
            child_desc = count_descendants(G, child)
            child_angle = angle_range * child_desc / total_desc

            mid_angle = current_angle + child_angle / 2
            r = depth * radius_step
            x = r * math.cos(mid_angle)
            y = r * math.sin(mid_angle)
            pos[child] = (x, y)

            layout_children(child, depth + 1, current_angle, current_angle + child_angle)
            current_angle += child_angle

    layout_children(root, 1, 0, 2 * math.pi)
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

    # Build networkx graph
    G = nx.DiGraph()
    for node_id, info in tree_data.items():
        G.add_node(node_id, formula=info["formula"], composite=info["composite"]) 
        if info["parent_id"] is not None:
            G.add_edge(info["parent_id"], node_id)

    # Find root
    root_id = next(nid for nid, info in tree_data.items() if info["parent_id"] is None)

    # Compute layout and flip 180 degrees
    pos = radial_layout(G, root_id, radius_step=3.0)
    pos = {nid: (-x, -y) for nid, (x, y) in pos.items()}

    # Collect values for coloring: composite, ehull_reward, r_dos
    composites = {}
    ehull_rewards = {}
    r_doss = {}
    for nid, info in tree_data.items():
        if info.get("composite") is not None:
            composites[nid] = info["composite"]
        # compute ehull_reward if e_hull present
        if info.get("e_hull") is not None:
            try:
                eh = ehull_reward(info["e_hull"]) if info["e_hull"] is not None else np.nan
            except Exception:
                eh = np.nan
            ehull_rewards[nid] = eh
        # r_dos available
        r_doss[nid] = info.get("r_dos", 0.0)

    # Prepare colormap helper
    def make_cmap_and_norm(values, cmap_name='RdBu'):
        arr = [v for v in values if v is not None and not pd.isna(v)]
        if not arr:
            return None, None
        vmin = min(arr)
        vmax = max(arr)
        cmap = cm.get_cmap(cmap_name)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        return cmap, norm

    # Create white->color colormaps for clear gradient visualization
    def white_to_color_cmap(color):
        return mcolors.LinearSegmentedColormap.from_list('wtc', ['white', color])

    cmap_comp = white_to_color_cmap('#1f77b4')
    arr_comp = [v for v in composites.values() if v is not None and not pd.isna(v)]
    norm_comp = mcolors.Normalize(vmin=min(arr_comp), vmax=max(arr_comp))

    cmap_ehull = white_to_color_cmap('#ff7f0e')
    arr_eh = [v for v in ehull_rewards.values() if v is not None and not pd.isna(v)]
    norm_ehull = mcolors.Normalize(vmin=min(arr_eh), vmax=max(arr_eh)) if arr_eh else None

    cmap_rdos = white_to_color_cmap('#2ca02c')
    arr_r = [v for v in r_doss.values() if v is not None and not pd.isna(v)]
    norm_rdos = mcolors.Normalize(vmin=min(arr_r), vmax=max(arr_r)) if arr_r else None

    if cmap_comp is None:
        print("No composite scores available")
        return 1

    # Prepare node colors for each metric
    def node_colors_for(metric_dict, cmap, norm):
        cols = []
        for nid in G.nodes():
            if nid in metric_dict and norm is not None:
                cols.append(cmap(norm(metric_dict[nid])))
            else:
                cols.append('lightgray')
        return cols

    node_colors_comp = node_colors_for(composites, cmap_comp, norm_comp)
    node_colors_ehull = node_colors_for(ehull_rewards, cmap_ehull, norm_ehull)
    # Scale r_dos by 2.5 for coloring (visual weighting), do not change label text
    rdos_scaled = {k: 2.5 * v for k, v in r_doss.items()}
    # use white->green cmap for scaled rdos
    cmap_rdos_scaled = cmap_rdos
    norm_rdos_scaled = mcolors.Normalize(vmin=min(rdos_scaled.values()), vmax=max(rdos_scaled.values())) if rdos_scaled else None
    node_colors_rdos = node_colors_for(rdos_scaled, cmap_rdos_scaled, norm_rdos_scaled)

    # Set global font size to 10pt for consistent publication text
    plt.rcParams.update({'font.size': 10})

    # Create a wide figure: 6in wide x 2.75in tall with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(6, 2.75), constrained_layout=True)

    panels = [
        (axes[0], node_colors_comp, cmap_comp, norm_comp, 'Composite Score'),
        (axes[1], node_colors_ehull, cmap_ehull, norm_ehull, r"$r_{E_{\mathrm{Hull}}}$"),
        (axes[2], node_colors_rdos, cmap_rdos_scaled, norm_rdos_scaled, r"$r_{\mathrm{DOS}}$")
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
