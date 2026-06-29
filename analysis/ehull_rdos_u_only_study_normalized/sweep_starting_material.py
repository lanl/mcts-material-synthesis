"""Starting-material sweep for this study's convergence_by_starting_material.png.

Gamma-normalized counterpart of analysis/ehull_rdos_u_only_study/
sweep_starting_material.py: every replicate also overrides gamma to
NORMALIZED_GAMMA (see generate_figures.py), so results land in their own
starting_material_sweep_normalized/ - the calibrated study's
starting_material_sweep/ is untouched.

Because gamma changes which compound is the true global-best U-only
composition (UTi6Sn6 here, vs. the calibrated study's UZr6Pb6 - see
compute_global_u_only_ranks in generate_figures.py), the calibrated study's
V/Ru/Pd/Cu6Ge6U ladder (chosen to land on d=2/4/6/8 to UZr6Pb6) no longer
lands on a clean d=2/4/6/8 spread to UTi6Sn6 (it gives d=1/5/7/7 instead -
see the d=1 case is barely informative for an edit-distance sweep, and the
d=7/7 tie collapses two of the four points). This sweep uses group_iv='Sn'
(matching the target's own group_iv exactly, so the move-graph distance is
governed purely by the transition-metal graph - see _edit_distance_to_target)
and picks transition metals that land exactly on d=2/4/6/8 to Ti (the
target's transition metal): Cr, Fe, Ni, Pt.

Reuses the shared replicate-running harness (run_sweep/save_sweep_results)
from sensitivity_studies/scripts/common.py - that module also backs the
other (generic MCTS-hyperparameter) sweeps there, so it stays put; only this
sweep is study-specific enough to live alongside generate_figures.py.
Output: sensitivity_studies/results/starting_material_sweep_normalized/convergence_data.csv
(path determined by common.py's own location, not this script's).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'sensitivity_studies' / 'scripts'))
from common import run_sweep, save_sweep_results, sweep_result_path

SWEEP_NAME = 'starting_material_sweep_normalized'

# 1 / (max raw r_DOS across the 108 U-only compounds) - matches
# generate_figures.py's NORMALIZED_GAMMA.
NORMALIZED_GAMMA = 1.0 / 2516.1664410449775

VALUES = {
    'Cr6Sn6U (d=2)': dict(transition_metal='Cr', group_iv='Sn', gamma=NORMALIZED_GAMMA),
    'Fe6Sn6U (d=4)': dict(transition_metal='Fe', group_iv='Sn', gamma=NORMALIZED_GAMMA),
    'Ni6Sn6U (d=6)': dict(transition_metal='Ni', group_iv='Sn', gamma=NORMALIZED_GAMMA),
    'Pt6Sn6U (d=8)': dict(transition_metal='Pt', group_iv='Sn', gamma=NORMALIZED_GAMMA),
}

if __name__ == '__main__':
    df = run_sweep(SWEEP_NAME, VALUES, checkpoint_path=sweep_result_path(SWEEP_NAME))
    save_sweep_results(df, SWEEP_NAME)
