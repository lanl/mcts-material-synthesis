# Architecture

## Scope

The repo currently implements the first executable stage of the proposal:

- ingest public synthesis corpora
- normalize mined records into a shared route schema
- retrieve analogous literature routes for a target
- generate candidate procedures with modality-aware grammars
- score completed routes with hard checks, heuristics, and a judge layer
- rank routes with Monte Carlo Tree Search
- generate retrospective benchmark splits and run baseline/ablation evaluation summaries

The planner now supports three executable planning modes:

- `solid_state`
- `hydrothermal`
- `precipitation`

## Active package

All active source code now lives in `synthesis_planner/`.

- `cli.py`
  - entrypoints for `download-data`, `prepare-data`, and `plan`
- `datasets.py`
  - downloads the public datasets
  - normalizes raw records into `RouteRecord` JSONL files
- `formula.py`
  - parses inorganic formulas
  - infers coarse target families
- `chemistry.py`
  - balances precursor sets to targets with common volatile products/reactants
  - infers coarse route-level oxidation/reduction demand
  - derives thermodynamic proxy features from the balanced reaction
- `schema.py`
  - shared dataclasses for routes, planning state, actions, and scores
- `retrieval.py`
  - finds analogous routes
  - builds precursor priors from literature usage
- `grammar.py`
  - modality-aware action expansion for solid-state, hydrothermal, and precipitation planning
- `scoring.py`
  - chemistry-aware route evaluation and deterministic judge output
- `constraints.py`
  - hard validity checks, lab-constraint gating, and redox/stoichiometry validation
- `judge.py`
  - pluggable retrieval-grounded judge interface
  - deterministic offline judge plus structured OpenAI-compatible judge
- `mcts.py`
  - compact PUCT-style tree search
- `planner.py`
  - top-level orchestration, baselines, and portfolio selection
- `benchmark.py`
  - split generation, baselines, ablations, and retrospective evaluation metrics

## Planning pipeline

1. Raw datasets are downloaded into `data/raw/`.
2. `prepare-data` converts them into normalized JSONL route files in `data/processed/`.
3. `plan` loads processed routes for the requested modality.
4. Retrieval finds analogous target recipes.
5. Precursor candidates are proposed from exact analog routes and element-level precursor usage.
6. MCTS explores a modality-aware grammar:
   - choose precursor set
   - choose preparation or solution-setup steps
   - choose heating / hydrothermal / precipitation actions
   - choose post-processing and finalization
7. Hard constraints gate route validity and emit explicit blocking flags.
8. Stoichiometric balancing and redox analysis convert terminal routes into explicit balanced-reaction and environment-consistency signals.
9. Thermodynamic proxy features are derived from the balanced route and combined with retrieval, condition, and practicality signals.
10. A judge interface evaluates route coherence with retrieved evidence.
   - offline deterministic rubric by default
   - optional model-backed structured judge for retrieval-grounded scoring
11. Rollouts complete partial routes and score them.
12. A small diverse top-k portfolio is written to `planning_results/`.
13. Optional benchmark tooling evaluates held-out targets under split strategies such as target-formula and chemical-system splits, plus baseline and ablation methods.

## Intentional simplifications

- No live LLM calls by default: the judge interface is pluggable, but the current shipped implementation is deterministic and rubric-based.
- No live thermodynamics APIs: the current chemistry layer uses offline balance/redox proxies rather than formation-energy or phase-diagram calls.
- The model-backed judge depends on an API key and a compatible model/endpoint configuration, so deterministic judging remains the default offline path.
- Benchmarking is still lightweight relative to the full proposal: it provides split generation and a small retrospective metric set, not the complete baseline/ablation framework yet.

## Repo cleanup

The old crystal-specific package and legacy study directories were removed to keep the repository aligned with the proposal:

- removed `mcts_crystal/`
- removed `analysis/`
- removed `sensitivity_studies/`
- removed `examples/`

That leaves the repo centered on the synthesis-planning package, its tests, and the proposal-driven data workflow.
