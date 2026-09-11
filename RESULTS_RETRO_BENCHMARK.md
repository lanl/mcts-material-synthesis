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

## Run C — analog-free vs analog-rich stratification (solid-state, chemical_system, leakage CLEAN)

Test targets split by retrieval support (similarity to the nearest train
analog): the **analog-free / novel** stratum (bottom third, support 0.07-2.0)
has no close recipe to copy; the **analog-rich** stratum (top third, support
~3.55) does. 60 targets per stratum.

| stratum | method | synth | op_sim | cond | stage | solve | rec@1 | jacc |
|---|---|---|---|---|---|---|---|---|
| novel | **mcts** | 0.69 | 0.39 | 0.28 | 0.30 | 0.85 | 0.00 | 0.08 |
| novel | nearest_neighbor | 0.50 | 0.45 | 0.38 | 0.37 | 0.88 | 0.00 | 0.21 |
| novel | frequency_prior | **0.72** | 0.40 | 0.35 | 0.35 | 0.83 | 0.00 | 0.07 |
| novel | random | 0.42 | 0.36 | 0.13 | 0.23 | 0.97 | 0.00 | 0.01 |
| rich | mcts | 0.91 | 0.54 | 0.37 | **0.58** | 0.98 | 0.15 | 0.46 |
| rich | nearest_neighbor | 0.86 | 0.54 | 0.31 | 0.38 | 0.97 | 0.03 | 0.53 |
| rich | frequency_prior | **0.93** | 0.50 | 0.44 | 0.42 | 1.00 | 0.30 | 0.58 |
| rich | random | 0.45 | 0.45 | 0.24 | 0.37 | 0.97 | 0.00 | 0.05 |

### What this shows (the clearest result so far)

1. **Nearest-neighbor collapses on novel targets** (synth 0.86 -> 0.50), exactly
   as predicted: recipe-copying fails with no close analog.
2. **MCTS holds up far better and clearly beats nearest-neighbor on novel
   targets** (synth 0.69 vs 0.50; drop of only 0.22 vs NN's 0.36). This is the
   real "search reasons from chemistry where retrieval can't" signal - MCTS's
   headline win over the retrieval baseline.
3. **But the frequency prior does NOT collapse and still marginally leads MCTS**
   even on novel targets (0.72 vs 0.69). Element-level precursor statistics
   transfer to novel compositions (a novel oxide still uses common oxide/
   carbonate precursors), so the frequency prior is the true baseline to beat -
   not retrieval.
4. **Why MCTS still trails the frequency prior: its condition search is *worse*
   than the prior's static defaults** (novel cond 0.28 vs 0.35). Since MCTS
   already inherits the frequency prior for precursors, the only way it beats the
   prior is by making conditions/stages *better* - and right now the condition
   search subtracts value.

### Actionable conclusion

The competitor is the **frequency prior**, and MCTS ties/loses to it only because
its **condition/stage search underperforms simple data-driven defaults**. Concrete
path to an outright MCTS win: (a) fix the hard-coded precipitation calcine
temperature and make all conditions data-driven from mined templates; (b) bias
the multi-stage prior toward the analog stage-count distribution; (c) reward
condition quality more strongly. Target metric: MCTS cond/stage > frequency
prior on the analog-free stratum, lifting novel synth above 0.72.

## Run D — after the three condition/stage fixes

Fixes applied: (1) data-driven solution calcine/anneal temperatures; (2)
stage-count continuation biased to the analog distribution; (3) graded,
non-saturating, data-driven condition reward (weight 0.8->1.3). **NB:** the
condition rewrite redefines the `synth` metric, so absolute synth is not
comparable to Runs A-C - only within-run gaps are.

Analog-free / analog-rich (solid-state, chemical_system, 60/stratum, leakage CLEAN):

| stratum | method | synth | op_sim | cond | stage |
|---|---|---|---|---|---|
| novel | mcts | 0.63 | 0.40 | 0.31 | 0.35 |
| novel | nearest_neighbor | 0.42 | 0.43 | 0.35 | 0.38 |
| novel | frequency_prior | **0.68** | 0.39 | 0.35 | 0.35 |
| rich | mcts | 0.85 | 0.52 | 0.38 | **0.60** |
| rich | nearest_neighbor | 0.76 | 0.51 | 0.29 | 0.40 |
| rich | frequency_prior | **0.86** | 0.51 | 0.40 | 0.43 |

Precipitation (standard, chemical_system, 60 targets, leakage CLEAN):

| method | synth | op_sim | cond | stage |
|---|---|---|---|---|
| mcts | 0.74 | 0.50 | **0.40** | 0.47 |
| nearest_neighbor | 0.44 | 0.65 | 0.26 | 0.50 |
| frequency_prior | 0.78 | 0.45 | 0.39 | 0.45 |

### What the fixes achieved

- **Precipitation conditions: cond 0.06 -> 0.40**, now *beating* the frequency
  prior (0.39) and nearest-neighbor (0.26). Fix #1 (data-driven calcine
  temperature) worked as intended.
- **Stage-count match on analog-rich: MCTS 0.60**, clearly best (freq 0.43, NN
  0.40). Fix #2 worked.
- **MCTS beats nearest-neighbor on synthesizability in every cell** (novel 0.63
  vs 0.42; precip 0.74 vs 0.44), confirming search dominates recipe-copying.

### What still did not clear the bar (and why)

- **On the novel stratum MCTS still trails the frequency prior on synth (0.63 vs
  0.68)**, and its novel condition score (0.31) is still <= the prior's (0.35).
- **Root cause is fundamental, not a tuning miss:** the fixes help *where there
  is data signal*. On analog-free targets there are, by definition, no close
  analogs, so data-driven conditions have nothing to drive from - the
  temperature signal vanishes exactly where novelty begins, and a sensible
  default (the frequency prior) stays marginally ahead. Heuristic, data-mined
  conditioning cannot beat a prior in the regime with no data.

### Implication

Beating the frequency prior **on novel targets** requires a signal that does not
come from analogs - i.e. a genuine **physics / forward model** (formation
energies, phase stability, a synthesizability predictor) that MCTS can optimize
where literature statistics run out. That is the next real lever; further
heuristic tuning has reached its ceiling on this corpus.

## Run E - after adding the offline physics (thermodynamic) signal

Added `core/physics.py`: balanced-reaction driving force from a curated
formation-enthalpy table + an oxide-sum estimator (novel coverage) + an MP/DFT
provider seam; folded (mean-centered) into the reward and the synth metric.
**NB:** synth now includes the physics term, so absolute synth is not comparable
to earlier runs - only within-run gaps are.

Analog-free / analog-rich (solid-state, chemical_system, 60/stratum, leakage CLEAN):

| stratum | method | synth | op_sim | cond | stage |
|---|---|---|---|---|---|
| novel | mcts | 0.61 | 0.40 | 0.33 | 0.38 |
| novel | nearest_neighbor | 0.43 | 0.43 | 0.35 | 0.38 |
| novel | frequency_prior | **0.64** | 0.39 | 0.35 | 0.33 |
| rich | mcts | 0.76 | 0.53 | 0.37 | **0.62** |
| rich | nearest_neighbor | 0.70 | 0.51 | 0.29 | 0.37 |
| rich | frequency_prior | **0.79** | 0.52 | 0.40 | 0.45 |

### Outcome

- **Physics did not flip the novel-target result.** MCTS still marginally trails
  the frequency prior on novel synth (0.61 vs 0.64; gap narrowed from 0.05 to
  0.03) and on analog-rich (0.76 vs 0.79). MCTS keeps beating nearest-neighbor
  everywhere and leads stage-count match (0.62 rich).
- **Why (as predicted):** the offline oxide-sum estimate is *neutral* for
  oxide->oxide reactions (it cannot see ternary stabilization), so on the
  mostly-oxide novel targets the physics term rarely fires; where it does
  (nitrate/oxalate precursor energetics) the frequency prior already captures it.

### Established ceiling and the remaining lever

Across Runs A-E, on this predominantly single-step, oxide-heavy corpus:
**MCTS reliably beats retrieval/nearest-neighbor, and beats the frequency prior
on procedure axes where data signal exists (precipitation conditions, stage
structure) - but ties/marginally trails the frequency prior on overall
synthesizability, including on novel targets, under both heuristic and offline-
physics rewards.** Beating the prior on novel targets requires a *real* stability
signal - Materials Project / DFT / ML formation energies via the (now wired)
`physics_provider` seam - or a change of objective (property optimization /
prospective validation) rather than literature-recall.
