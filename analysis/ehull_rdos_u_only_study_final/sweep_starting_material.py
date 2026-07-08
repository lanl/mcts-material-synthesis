"""Starting-material sweep for this study's convergence_by_starting_material.png.

Final-parameter counterpart combining both parameter changes:
  - gamma = NORMALIZED_GAMMA (1/max raw r_DOS = 1/2516.1664410449775)
  - rollout_aggregation = 'mean'

The global-best U-only compound under NORMALIZED_GAMMA is UTi6Sn6 (not UZr6Pb6
as in the calibrated study), so the same Cr/Fe/Ni/Pt6Sn6U d=2/4/6/8 ladder
as analysis/ehull_rdos_u_only_study_normalized/sweep_starting_material.py is
reused here (group_iv='Sn' to match the target's own group_iv exactly, so the
move-graph distance is governed purely by the transition-metal graph).

Output: sensitivity_studies/results/starting_material_sweep_final/convergence_data.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'sensitivity_studies' / 'scripts'))
from common import run_sweep, save_sweep_results, sweep_result_path

SWEEP_NAME = 'starting_material_sweep_final'

NORMALIZED_GAMMA = 1.0 / 2516.1664410449775

VALUES = {
    'Cr6Sn6U (d=2)': dict(transition_metal='Cr', group_iv='Sn', gamma=NORMALIZED_GAMMA, rollout_aggregation='mean'),
    'Fe6Sn6U (d=4)': dict(transition_metal='Fe', group_iv='Sn', gamma=NORMALIZED_GAMMA, rollout_aggregation='mean'),
    'Ni6Sn6U (d=6)': dict(transition_metal='Ni', group_iv='Sn', gamma=NORMALIZED_GAMMA, rollout_aggregation='mean'),
    'Pt6Sn6U (d=8)': dict(transition_metal='Pt', group_iv='Sn', gamma=NORMALIZED_GAMMA, rollout_aggregation='mean'),
}

if __name__ == '__main__':
    df = run_sweep(SWEEP_NAME, VALUES, checkpoint_path=sweep_result_path(SWEEP_NAME))
    save_sweep_results(df, SWEEP_NAME)
