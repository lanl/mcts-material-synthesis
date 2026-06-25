"""Sensitivity sweep: secondary MCTS-loop parameters (n_rollout, rollout_depth,
termination_limit), each varied one-at-a-time against the calibrated baseline
(n_rollout=1, rollout_depth=3, termination_limit=25). These are separate from
the reward formula itself - distinct sub-sweep, CSV, and figure per parameter.
"""

from common import run_sweep, save_sweep_results

SWEEP_NAME = 'rollout_params_sweep'

N_ROLLOUT_VALUES = {
    '1 (calibrated)': dict(n_rollout=1),
    '3': dict(n_rollout=3),
    '5': dict(n_rollout=5),
}

# rollout_depth only affects the "extra" rollout samples beyond the first
# (depth=0) one - with n_rollout=1 (the calibrated value) there are none, so
# rollout_depth is a complete no-op there. To actually exercise it, this
# sub-sweep holds n_rollout=3 instead of the calibrated 1.
ROLLOUT_DEPTH_VALUES = {
    '1': dict(rollout_depth=1, n_rollout=3),
    '3 (calibrated)': dict(rollout_depth=3, n_rollout=3),
    '5': dict(rollout_depth=5, n_rollout=3),
}

TERMINATION_LIMIT_VALUES = {
    '10': dict(termination_limit=10),
    '25 (calibrated)': dict(termination_limit=25),
    '50': dict(termination_limit=50),
}

if __name__ == '__main__':
    df = run_sweep(f'{SWEEP_NAME}/n_rollout', N_ROLLOUT_VALUES)
    save_sweep_results(df, SWEEP_NAME, filename='convergence_data_n_rollout.csv')

    df = run_sweep(f'{SWEEP_NAME}/rollout_depth', ROLLOUT_DEPTH_VALUES)
    save_sweep_results(df, SWEEP_NAME, filename='convergence_data_rollout_depth.csv')

    df = run_sweep(f'{SWEEP_NAME}/termination_limit', TERMINATION_LIMIT_VALUES)
    save_sweep_results(df, SWEEP_NAME, filename='convergence_data_termination_limit.csv')
