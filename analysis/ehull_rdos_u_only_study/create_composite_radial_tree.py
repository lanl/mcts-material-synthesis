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

    # Collect composite scores for coloring
    composites = {}
    for nid, info in tree_data.items():
        if info["composite"] is not None:
            composites[nid] = info["composite"]

    all_comp = [v for v in composites.values()]
    if not all_comp:
        print("No composite scores available")
        return 1

    min_comp = min(all_comp)
    max_comp = max(all_comp)
    print(f"  Composite range: [{min_comp:.4f}, {max_comp:.4f}]")

    # Build blue-red colormap (blue=high/better, red=low/worse)
    colors_red = ['darkred', 'red', 'lightcoral', 'white']   # low to mid
    colors_blue = ['white', 'lightblue', 'blue', 'darkblue']  # mid to high
    n_bins = 100

    if min_comp < 0 and max_comp > 0:
        neg_range = abs(min_comp)
        pos_range = abs(max_comp)
        total = neg_range + pos_range
        n_neg = max(1, int(n_bins * neg_range / total))
        n_pos = max(1, n_bins - n_neg)
        cmap_low = mcolors.LinearSegmentedColormap.from_list('red', colors_red, N=n_neg)
        cmap_high = mcolors.LinearSegmentedColormap.from_list('blue', colors_blue, N=n_pos)
        colors_arr = np.vstack((cmap_low(np.linspace(0, 1, n_neg)),
                                cmap_high(np.linspace(0, 1, n_pos))))
        cmap = mcolors.LinearSegmentedColormap.from_list('red_blue', colors_arr)
    elif max_comp <= 0:
        cmap = mcolors.LinearSegmentedColormap.from_list('red', colors_red, N=n_bins)
    else:
        cmap = mcolors.LinearSegmentedColormap.from_list('blue', colors_blue, N=n_bins)

    norm = mcolors.Normalize(vmin=min_comp, vmax=max_comp)

    # Assign colors to nodes (root colored by its composite score like all others)
    node_colors = []
    for nid in G.nodes():
        if nid in composites:
            node_colors.append(cmap(norm(composites[nid])))
        else:
            node_colors.append('lightgray')

    # Create figure — 3.25in x 3.25in, single panel with horizontal colorbar below
    fig, ax_tree = plt.subplots(figsize=(3.25, 3.25))

    # Radial tree
    nx.draw(G, pos, ax=ax_tree, with_labels=False,
            node_color=node_colors, edge_color='gray', node_size=120,
            arrows=True, connectionstyle='arc3,rad=0.1',
            edgecolors='black', linewidths=0.8)
    ax_tree.axis('equal')

    # Horizontal colorbar below the tree
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_tree, orientation='horizontal',
                        fraction=0.05, pad=0.02, aspect=30)
    cbar.set_label('Composite Score', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Save
    output_path = script_dir / 'radial_tree_composite.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    return 0


if __name__ == '__main__':
    exit(main())
