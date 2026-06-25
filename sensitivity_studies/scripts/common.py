"""Shared setup/helpers for the MCTS sensitivity studies.

Each sweep script varies exactly one hyperparameter (or a small group of
closely related ones) away from the calibrated baseline below, runs N_SEEDS
replicate MCTS searches per value, and writes a long-format CSV of
best_reward (the composite reward, beta*ehull_reward + gamma*r_DOS) vs.
iteration to results/<sweep_name>/. plot_sweep.py then turns that CSV into a
convergence figure.

beta, gamma, the ehull_reward sharpness constant (120), the E_hull stability
threshold (0.05), and the rDOS Gaussian width sigma (0.5) are physics-informed
and are NOT swept here - only MCTS-algorithm hyperparameters are.
"""

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read

REPO_ROOT = Path(__file__).resolve().parents[2]

from mcts_crystal.node import MCTSTreeNode
from mcts_crystal.mcts import MCTS
from mcts_crystal.energy_calculator import MaceEnergyCalculator
from mcts_crystal.doscar_utils import DoscarRewardLookup
from mcts_crystal.cli import override_composition

# Calibrated baseline (matches the repo's current config.json). Each sweep
# overrides only the key(s) it is studying.
BASELINE = dict(
    transition_metal='Cr',
    group_iv='Sn',
    f_block_mode='u_only',
    selection_mode='ucb1',
    exploration_constant=0.5,
    epsilon=0.2,
    temperature=1.0,
    termination_limit=25,
    rollout_depth=3,
    n_rollout=1,
    rollout_method='ehull_rdos',
    beta=1.0,
    gamma=0.0001,
)

N_SEEDS = 5
N_ITERATIONS = 500
SEEDS = list(range(N_SEEDS))


def load_shared_resources():
    """Load the starting-structure template and the (seed-independent,
    reusable) energy/rDOS lookups once, to be shared across every replicate
    run in a sweep."""
    atoms_template = read(str(REPO_ROOT / 'examples' / 'mat_Pb6U1W6_sg191.cif'))
    energy_calc = MaceEnergyCalculator(
        csv_file=str(REPO_ROOT / 'high_throughput_mace_results.full.csv'),
        mp_api_key=None,
    )
    doscar_lookup = DoscarRewardLookup(
        peaks_file=str(REPO_ROOT / 'doscar_peaks_data_with_U.csv')
    )
    return atoms_template, energy_calc, doscar_lookup


def run_replicate(atoms_template, energy_calc, doscar_lookup, seed, params):
    """Run a single fresh MCTS search (params overrides BASELINE) and return
    its best_reward-vs-iteration array (length N_ITERATIONS+1, including the
    iteration-0 pre-search sentinel)."""
    cfg = {**BASELINE, **params}

    random.seed(seed)
    np.random.seed(seed)

    atoms = override_composition(
        atoms_template.copy(),
        transition_metal=cfg['transition_metal'],
        group_iv=cfg['group_iv'],
    )
    root = MCTSTreeNode(
        atoms, f_block_mode=cfg['f_block_mode'],
        exploration_constant=cfg['exploration_constant'],
        termination_limit=cfg['termination_limit'],
    )
    root.e_form, root.e_above_hull = energy_calc.calculate_energies(atoms)

    mcts = MCTS(root, epsilon=cfg['epsilon'], temperature=cfg['temperature'])
    mcts.run(
        n_iterations=N_ITERATIONS,
        energy_calculator=energy_calc,
        rollout_depth=cfg['rollout_depth'],
        n_rollout=cfg['n_rollout'],
        selection_mode=cfg['selection_mode'],
        rollout_method=cfg['rollout_method'],
        beta=cfg['beta'],
        gamma=cfg['gamma'],
        doscar_lookup=doscar_lookup,
    )
    return np.array(mcts.max_reward_history)


def run_sweep(sweep_name, param_overrides_by_value):
    """Run every (value, seed) replicate for a sweep and return a tidy
    long-format DataFrame: columns = [value, seed, iteration, best_reward].

    Args:
        sweep_name: used only for the progress log
        param_overrides_by_value: dict mapping a human-readable value label
            (e.g. "0.5" or "Cr6Sn6U") to the dict of BASELINE overrides that
            produce it (e.g. {'exploration_constant': 0.5})
    """
    atoms_template, energy_calc, doscar_lookup = load_shared_resources()

    rows = []
    t0 = time.time()
    for value_label, overrides in param_overrides_by_value.items():
        for seed in SEEDS:
            ts = time.time()
            history = run_replicate(atoms_template, energy_calc, doscar_lookup, seed, overrides)
            print(f"[{sweep_name}] value={value_label!r} seed={seed} "
                  f"-> {len(history)} pts, final_best={history[-1]:.4f} ({time.time()-ts:.1f}s)")
            for it, reward in enumerate(history):
                rows.append((value_label, seed, it, reward))
    print(f"[{sweep_name}] total runtime: {time.time()-t0:.1f}s")

    return pd.DataFrame(rows, columns=['value', 'seed', 'iteration', 'best_reward'])


def save_sweep_results(df, sweep_name, filename='convergence_data.csv'):
    out_dir = REPO_ROOT / 'sensitivity_studies' / 'results' / sweep_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    return out_path
