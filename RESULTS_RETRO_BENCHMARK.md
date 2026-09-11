# Retrosynthesis-style benchmark — leakage-controlled results

Harness: `synthesis_planner/retro_benchmark.py` (run via `scripts/run_retro_benchmark.py`).
All artifacts (stock, templates, retrieval index, precursor-frequency prior) are
mined **train-only**; a split-key audit asserts no held-out key leaks into train.
Deterministic judge only (no LLM/API).

**Metrics.** Headline = procedure quality: **synth** (offline synthesizability
proxy = mean of stoich / thermo / condition / stock-coverage of the produced
route), **op_sim** (operation-sequence similarity vs the closest gold route),
**cond** (calcination T within the gold envelope ±75 °C), **stage** (predicted
firing-stage count matches a gold route). Buildability = **solve** (route bottoms
out entirely in the train stock). Secondary diagnostic = **rec@k** (exact
precursor-set recovery) / **jacc** (soft precursor overlap).

## Run A — solid-state, held out by chemical system (60 targets, leakage CLEAN)

| method | synth | op_sim | cond | stage | solve | rec@1 | rec@3 | jacc |
|---|---|---|---|---|---|---|---|---|
| mcts | 0.82 | 0.52 | 0.35 | 0.27 | 0.92 | 0.13 | 0.13 | 0.38 |
| nearest_neighbor | 0.78 | 0.57 | 0.40 | 0.45 | 0.97 | 0.05 | 0.05 | 0.52 |
| frequency_prior | **0.84** | 0.48 | 0.35 | 0.40 | 0.92 | **0.25** | 0.27 | 0.53 |
| random | 0.39 | 0.46 | 0.18 | 0.25 | 0.87 | 0.00 | 0.00 | 0.03 |

## Run B — precipitation (depth modality), held out by chemical system (60 targets, leakage CLEAN)

| method | synth | op_sim | cond | stage | solve | rec@1 | rec@3 | jacc |
|---|---|---|---|---|---|---|---|---|
| mcts | 0.77 | 0.44 | 0.06 | 0.32 | 0.88 | 0.07 | 0.07 | 0.21 |
| nearest_neighbor | 0.44 | **0.65** | 0.26 | 0.50 | 0.80 | 0.02 | 0.03 | 0.16 |
| frequency_prior | **0.80** | 0.45 | 0.20 | 0.45 | 0.90 | 0.07 | 0.10 | 0.24 |
| random | 0.22 | 0.56 | 0.17 | 0.37 | 0.80 | 0.00 | 0.00 | 0.01 |

## Honest conclusion

**The reframe made MCTS competitive on synthesizability but did NOT produce a win
over the frequency prior — on either modality.**

- **synth:** MCTS ≈ frequency_prior (0.82 vs 0.84; 0.77 vs 0.80), both clearly
  above nearest_neighbor. So MCTS produces buildable, chemically-reasonable
  procedures — but no better than a one-line frequency baseline.
- **precursor recovery:** the frequency-prior seeding did not move MCTS's recall
  (0.13 / 0.07); the strong, simple frequency prior remains ahead or tied.
- **conditions / stage structure:** nearest_neighbor wins by copying whole
  literature recipes. MCTS is *weakest* on precipitation conditions (cond 0.06 —
  its calcination temperature is a hard-coded 500 °C, not data-driven) and on
  firing stage-count (its multi-stage MDP over-explores stage counts vs the
  mostly-single-stage literature).

**Root cause — this is a formulation limit, not a tuning bug.** On a corpus whose
"answer" is *use the common precursors under standard conditions*, the task is
retrieval/frequency-dominated. A frequency prior captures nearly all the
signal, so a large combinatorial search has almost nothing to add. MCTS's
strength — searching a huge space under a non-trivial objective — is not what
this task rewards.

## Two concrete, fixable MCTS weaknesses this surfaced

1. **Precipitation calcination temperature is hard-coded (500 °C)** in
   `grammar._solution_postprocess_actions`, giving cond=0.06. Make it data-driven
   from analogs / mined templates (as solid-state heating already is).
2. **Firing stage-count over-exploration** — bias the multi-stage priors toward
   the analog stage-count distribution (mostly 1) so MCTS stops proposing extra
   stages the literature does not use.

Fixing these would raise MCTS on cond/stage toward nearest_neighbor, but is
unlikely to beat the frequency prior on the headline — the ceiling here is set by
the task, not the search.

## Strategic implication for "MCTS for materials synthesis"

To make MCTS genuinely *win*, the objective must be one retrieval/frequency
**cannot** solve by construction:

1. **Novel / analog-free targets** — evaluate only on held-out targets with no
   close analog in train (low retrieval support), where copying/frequency fail
   and search must reason from chemistry. This is where MCTS should separate.
2. **Property/synthesizability optimization**, not literature imitation — reward a
   forward model of phase purity / yield / cost and ask MCTS to *optimize* it,
   scoring against that objective rather than recipe recall. (Needs a forward
   model or the A-Lab-style outcome labels the corpus lacks.)
3. **Genuinely multi-step chemistries** — flux/precursor-derived/MOF-derived
   routes with real depth, rather than the ~99%-single-step bulk oxide corpus.

The current benchmark, harness, leakage control, and reward are sound and
reusable; the open scientific question is choosing an objective where search
beats a prior.
