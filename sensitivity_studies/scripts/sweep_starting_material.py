"""Sensitivity sweep: starting material (transition_metal/group_iv override).

Four non-target compositions spanning a clean ladder of graph-distance (in
the substitution-move graph defined in mcts_crystal/node.py) to the nearest
of the four experimentally-synthesized target compounds (U-Sn-V, U-Sn-Nb,
U-Ge-Cr, U-Ge-Co): max one-hop distance to the farthest target is 3, 4, 5, 6
respectively. Cr6Sn6U (max=3) is the calibrated starting material in
config.json.
"""

from common import run_sweep, save_sweep_results

SWEEP_NAME = 'starting_material_sweep'

VALUES = {
    'Cr6Sn6U (max dist=3, calibrated)': dict(transition_metal='Cr', group_iv='Sn'),
    'V6Ge6U (max dist=4)': dict(transition_metal='V', group_iv='Ge'),
    'Nb6Ge6U (max dist=5)': dict(transition_metal='Nb', group_iv='Ge'),
    'Zr6Ge6U (max dist=6)': dict(transition_metal='Zr', group_iv='Ge'),
}

if __name__ == '__main__':
    df = run_sweep(SWEEP_NAME, VALUES)
    save_sweep_results(df, SWEEP_NAME)
