# Retrosynthesis-style benchmark — first leakage-controlled results

Harness: `synthesis_planner/retro_benchmark.py` (run via `scripts/run_retro_benchmark.py`).
All artifacts (stock, templates, retrieval index) are mined **train-only**; a
split-key audit asserts no held-out key leaks into train. Deterministic judge
only (no LLM/API).

## Run 1 — solid-state, held out by chemical system

- Split: `chemical_system`, test fraction 0.2, seed 0
- **Leakage audit: CLEAN** (no held-out chemical system appears in train)
- 60 target materials evaluated; **493 targets skipped** as un-parseable
  (solid-solution / variable-stoichiometry formulas the balancer cannot handle —
  a known data-quality limitation)
- MCTS: 60 iterations, 3 rollouts; top-k = 3; stock `min_frequency` = 20 (480 formulas)

| method | solve_rate | stock_cov | recall@1 | recall@3 | class_recall@k | prec_jaccard | cond_in_range |
|---|---|---|---|---|---|---|---|
| mcts | 0.92 | 0.93 | 0.13 | 0.13 | 0.18 | 0.38 | 0.35 |
| nearest_neighbor | 0.97 | 0.98 | 0.05 | 0.05 | 0.17 | 0.52 | 0.40 |
| frequency_prior | 0.92 | 0.93 | **0.25** | **0.27** | 0.02 | 0.53 | 0.35 |
| random | 0.87 | 0.90 | 0.00 | 0.00 | 0.30 | 0.03 | 0.18 |

(Metrics: **solve_rate** = top route bottoms out entirely in the train-mined
stock; **recall@k** = any of the top-k predicted precursor sets exactly matches
any gold literature route for that target; **class_recall@k** = precursor-class
multiset match; **prec_jaccard** = best formula-overlap of the top prediction;
**cond_in_range** = predicted calcination T within the gold temperature envelope
±75 °C.)

## Honest interpretation

1. **Solve-rate is saturated (~0.9 for every method) and therefore not
   discriminative here.** With a broad 480-precursor stock and ~99%-single-step
   solid-state chemistry, almost any reasonable proposal "solves." Solve-rate
   only becomes informative with a *stricter* stock (smaller `min_frequency` /
   vendor-catalog stock) or on harder (multi-cation, recursion-requiring)
   targets. This is the inorganic analog of retrosynthesis being "easy" when the
   building-block set is large.

2. **On exact precursor recovery, MCTS beats nearest-neighbor (0.13 vs 0.05) but
   loses to the simple frequency prior (0.25).** This is the "MCTS ≈ retrieval"
   risk made concrete: the dominant signal on held-out chemistries is "use the
   most common precursors for these elements," which the frequency prior encodes
   directly and which MCTS currently dilutes by searching a broader space.

3. **Nearest-neighbor wins on soft overlap and conditions** (jaccard 0.52, cond
   0.40) because it copies whole literature recipes; it loses on *exact* recovery
   because the nearest analog is rarely the exact held-out recipe.

4. The `random` baseline (scoring random train routes) lands at solve 0.87 /
   recall 0.00 — a sane lower bound confirming the recovery metrics are
   measuring something real.

## What this says about the objective

The retrosynthesis framing, harness, stock/solved criterion, and leakage control
all work and produce defensible numbers. But **search does not yet pay off over a
frequency prior on precursor recovery.** Closing that gap is the core remaining
scientific task, not more plumbing.

## Prioritized next steps

1. **Make solve-rate discriminative**: sweep stock `min_frequency` (e.g. 50/100)
   and report solve-rate sensitivity; consider a curated vendor-style stock.
2. **Make search earn its keep**: tune the reward so MCTS concentrates on the
   high-frequency precursor region it already has access to (it should *dominate*
   the frequency prior, not trail it); consider seeding the value with the
   frequency prior and letting search refine conditions/stages.
3. **Add a beam-search baseline** — the decisive "is it search or just the
   candidate set?" control (design doc WS-C).
4. **Run the `target_formula` split and the solution modality** (where #2 gives
   genuine depth-2 DAGs) and report per-split.
5. **Report multi-seed bootstrap CIs** for recall@k so MCTS−frequency_prior has
   an error bar.
