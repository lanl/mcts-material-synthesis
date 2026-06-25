"""Sensitivity sweep: starting material (transition_metal/group_iv override).

Four compositions spanning a ladder of move-graph edit distance (in the
substitution-move graph defined in mcts_crystal/node.py) to the true global-
best U-only compound (UZr6Pb6, see compute_global_u_only_ranks in
generate_figures.py): d=2, 4, 6, 8 out of a max possible d=9. This spread
shows convergence speed degrading as the starting point gets farther from
the optimum.
"""

from common import run_sweep, save_sweep_results

SWEEP_NAME = 'starting_material_sweep'

VALUES = {
    'V6Ge6U (d=2)': dict(transition_metal='V', group_iv='Ge'),
    'Ru6Ge6U (d=4)': dict(transition_metal='Ru', group_iv='Ge'),
    'Pd6Ge6U (d=6)': dict(transition_metal='Pd', group_iv='Ge'),
    'Cu6Ge6U (d=8)': dict(transition_metal='Cu', group_iv='Ge'),
}

if __name__ == '__main__':
    df = run_sweep(SWEEP_NAME, VALUES)
    save_sweep_results(df, SWEEP_NAME)
