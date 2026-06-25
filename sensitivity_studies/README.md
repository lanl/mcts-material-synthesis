# MCTS Sensitivity Studies

Replicate-run sensitivity studies for the U-only `ehull_rdos` search, used to
"calibrate" MCTS-algorithm hyperparameters (not the reward formula itself -
the ehull_reward sharpness constant (120), the E_hull stability threshold
(0.05), and the rDOS Gaussian width sigma (0.5) are physics-informed and are
**not** swept here).

## Methodology

Every sweep holds the calibrated baseline (matching `config.json`: U-only
f-block mode, `ucb1` selection, `exploration_constant=0.5`, `Cr6Sn6U` starting
material, `ehull_rdos` reward with `beta=1.0`/`gamma=0.0001`,
`termination_limit=25`, `rollout_depth=3`, `n_rollout=1`) and varies exactly
one parameter (or a small closely-related group of parameters) away from it.
For each value tested, 5 replicate seeds are run for 500 iterations each (all
compounds in this search space are already in `high_throughput_mace_results.full.csv`,
so this is fast - no live MACE/Materials Project calls). The metric is the
running-best composite reward per iteration, averaged across the 5 seeds with
a shaded +/-1 std-dev band, on a log-scale iteration axis (convergence in
this search space happens within the first ~10-50 iterations, so a linear
axis compresses all the interesting dynamics into an unreadable sliver).

Run `bash scripts/run_all.sh` to reproduce everything (~15-20 min).

## Sweeps and findings

### `c_sweep` - exploration_constant (0.05, 0.1, 0.2, 0.5 calibrated, 1.0)
**No measurable effect.** See "Why c and selection_mode show no effect" below.

### `starting_material_sweep` - Cr6Sn6U / V6Ge6U / Nb6Ge6U / Zr6Ge6U
(graph-distance-to-nearest-target of 3/4/5/6 hops respectively, excluding the
four target compounds themselves)

**Clear, monotonic effect**: convergence speed tracks starting-point distance
directly - Cr6Sn6U (distance 3, the calibrated choice) reaches the plateau
fastest, Zr6Ge6U (distance 6) slowest. This is the parameter that actually
matters most for "rate of convergence" in this search space.

### `selection_mode_sweep` - ucb1 (calibrated) / epsilon_greedy / boltzmann / puct / hybrid
**No measurable effect**, for the same root cause as `c_sweep`.

### `rollout_params_sweep` - n_rollout, rollout_depth, termination_limit (one-at-a-time)
- **n_rollout** (1 calibrated, 3, 5): real effect - more rollout samples per
  expansion gives a higher, less noisy reward estimate immediately at
  iteration 1 (`reward = max` over samples), so n_rollout=3/5 start closer to
  the eventual plateau than n_rollout=1.
- **rollout_depth** (1, 3 calibrated, 5; n_rollout fixed at 3 here since
  rollout_depth only affects the "extra" rollout samples beyond the first,
  which don't exist when n_rollout=1): smaller, real effect in the same
  direction as n_rollout.
- **termination_limit** (10, 25 calibrated, 50): **no effect on the reward
  curve** (same root cause as `c_sweep`/`selection_mode_sweep` - the optimum
  is found well before termination ever triggers), but a strong, expected
  effect on how long the search runs before halting: mean iterations
  completed (of a 500-iteration budget) is 144 (limit=10), 407 (limit=25),
  500/never-terminates (limit=50) - see `iterations_vs_termination_limit.png`.

## Why `c` and `selection_mode` show no effect

This isn't a bug - it follows directly from how expansion works in
`mcts_crystal/node.py`/`mcts.py`. A node's `expandable` flag stays `True`
until **every** child in its `expansion_list` has been added to the tree via
a uniform-random draw (`expansion_simulation`'s `random.choice` over the
remaining expansion list) - and `selection_mode`/`exploration_constant` only
govern the choice *among already-expanded children* in `select_node()`,
which only runs once a node is no longer `expandable`. With only 8 immediate
moves from any node in this u_only search space (4 transition-metal options x
2 Group IV chain-neighbors x 1 f-block option), and the global optimum
reachable within 1-2 hops of the calibrated starting material, the optimum
is discovered purely through this mode-independent exhaustion phase, before
`selection_mode`/`c`-dependent selection ever gets a chance to matter.
`starting_material` and `n_rollout`/`rollout_depth` show real effects
precisely because they act *during* (or before) that same early phase, not
after it.

## Files

```
sensitivity_studies/
  scripts/
    common.py                     # shared setup + replicate-running helper
    sweep_c.py
    sweep_starting_material.py
    sweep_selection_mode.py
    sweep_rollout_params.py        # n_rollout, rollout_depth, termination_limit
    plot_sweep.py                  # generic convergence-curve figure (3x3in, 10pt)
    plot_termination_iterations.py # iterations-completed bar chart for termination_limit
    run_all.sh                      # reproduces everything
  results/
    c_sweep/
    starting_material_sweep/
    selection_mode_sweep/
    rollout_params_sweep/
```

Each `results/<sweep>/` directory has a `convergence_data*.csv` (long-format:
`value, seed, iteration, best_reward`) and the corresponding `.png` figure(s).
