# Migration Plan: modularizing `mcts-material-synthesis`

Goal: bring this repo up to the modular architecture and packaging conventions of
the reference framework `mcts-materials` **without losing any synthesis-specific
behavior**. The existing pytest suite (89 passed, 1 skipped — the skip needs a
Materials Project API key) is the behavioral contract and must stay green.

## 1. The core architectural tension and the decision

Synthesis planning builds a **route** — an ordered sequence of actions/stages
(`precursors → preparation → heating → finalize`, with modality variants). It
operates on an immutable `PlanningState` through `grammar.expand_state /
apply_action / rollout_completion`, and its evaluation produces a rich
`PlannedRoute` object (score breakdown + hard checks + thermo + judge) that is
accumulated across the whole search for portfolio/diversity selection.

The reference framework's generic `core` is built around a different problem:
**mutate a single `Material`**. Its contracts are:
- `Material.get_identifier()` for dedup (each material appears once in the tree);
- `MoveGenerator.generate_moves(material) -> List[Material]` (neighbors of one material);
- `PropertyEvaluator.evaluate(material) -> Dict[str, float]` (a scalar property map);
- `RewardFunction.compute_reward(props) -> float`;
- an **async** `MCTS` that expands one child per iteration and probes reward via
  max-along-walk rollouts, keeping no terminal objects.

These do not fit route planning:
1. **Actions are context-conditioned.** `expand_state` needs `analogs` and
   `candidate_precursor_sets` (retrieval context), not just a parent object.
2. **Evaluation returns a rich object, not a scalar map.** The `PlannedRoute`
   (with `ScoreBreakdown`, `HardCheckResult`, `ThermoAnalysisResult`,
   `JudgeResult`) is the product; a `Dict[str, float]` + scalar reward throws away
   everything the CLI, portfolio selection, and benchmark need.
3. **Terminal objects are accumulated.** `SynthesisPlanner` selects a top-k
   *diverse portfolio* over all terminal routes seen; the generic MCTS keeps none.
4. **Dedup is wrong here.** Reaching the same intermediate state by different
   paths and accumulating many terminal routes is desired, not deduplicated.
5. **Sync vs async.** The judge (LLM) and Materials Project calls are synchronous
   and already batched inside a single `evaluate_state`; the async engine buys
   nothing and would complicate the judge path.

**Decision — pragmatic hybrid (option 3).** Keep the bespoke **sync PUCT engine**
(`MonteCarloTreeSearch`/`TreeNode`) as the synthesis domain's core, and adopt the
reference's *packaging, layout, layering, and tooling* conventions around it. We
do **not** rebuild synthesis on the `Material`/`MoveGenerator`/async-`MCTS`
interfaces — that would force route planning into an ill-fitting shape and break
the behavioral contract for no benefit. This is exactly the "do not force
uniformity" guidance in the brief.

## 2. What we adopt from the reference

- **src-layout**: `synthesis_planner/` → `src/synthesis_planner/`; pyproject
  `package-dir = {"" = "src"}`.
- **Layered sub-packages** mirroring the reference's core→domain→cli split, chosen
  to respect the *actual* module dependency DAG (see below):
  - `core/` — the route-planning engine and immutable domain model + evaluation:
    `schema, formula, chemistry, constraints, grammar, judge, scoring, mcts`.
    These modules import only each other, so they form a self-contained core.
  - `data/` — ingestion / retrieval / external APIs (heavy deps lazy):
    `datasets, retrieval, materials_project`.
  - top level (orchestration): `planner, benchmark, failure_taxonomy,
    judge_calibration`.
  - `cli/` — `main.py` (the argparse app), with `cli/__init__.py` re-exporting
    `build_parser, load_config, main` so `synthesis_planner.cli` still works.
- **pyproject conventions**: `wheel` in build-system; LANL metadata (license,
  authors, keywords, classifiers, urls); heavy/optional deps moved to **extras**
  (`judge = openai`, `mp = mp-api`) with `dev` pulling in `judge` so `.[dev]`
  remains sufficient for the full test suite; `all` aggregating everything;
  reference-style `[tool.pytest.ini_options]`, `[tool.black]`, `[tool.ruff]`,
  `[tool.mypy]`.
- **Lazy heavy imports** (already true, preserved): `openai` inside `judge.py`,
  `mp_api` inside `materials_project.py`.
- **LANL copyright header** on the new package `__init__.py` files.

### Why the dependency graph forbids a deeper split
`scoring → judge`, and `benchmark ↔ failure_taxonomy` is a genuine (partly lazy)
cycle. Splitting `judge` or `failure_taxonomy` into their own packages would turn
these into cross-package import cycles. Keeping `judge` inside `core` (its only
deps are `formula`+`schema`) and keeping `failure_taxonomy` beside `benchmark`
keeps every package boundary acyclic: `core` ← `data` ← orchestration ← `cli`.

## 3. Deliberate divergences (kept, with justification)

- **CLI stays argparse, not typer.** `tests/test_cli.py` asserts directly on the
  argparse parser (`build_parser({})`, `parser.parse_args(...).command`), and the
  CLI's config-file-driven defaults (`config.py`/`config.json` merged into
  argparse defaults) are a feature typer does not naturally provide. Converting
  would delete coverage and behavior for pure cosmetic uniformity. The
  `mcts-plan` entry point and `run_mcts.py` wrapper are preserved.
- **Config stays dict-based, not pydantic.** `load_config` merges
  `config.py`/`config.json` and carries an open-ended `judge_config` passthrough
  dict; the test suite locks this shape. A pydantic `Config` would add a parallel,
  unused validation layer. Considered and rejected; revisit if a typed config is
  later desired.

## 4. Execution steps

1. Branch `modularize` off `develop` (done).
2. Write this plan; commit.
3. `git mv` modules into `src/synthesis_planner/{core,data,cli}/` + top level; add
   `__init__.py` for each new package (with LANL header + re-exports).
4. Rewrite intra-package imports for moved modules (core bodies need no change;
   `data`, orchestration, and `cli` get updated relative paths).
5. Rewrite test + conftest import paths to the new module locations.
6. Update `pyproject.toml` (src-layout, packages, extras, tooling).
7. Update `CLAUDE.md` / `README.md` to the new structure; keep `run_mcts.py`.
8. Reinstall `.[dev]`, run pytest, confirm 89 passed / 1 skipped (same as baseline).

## 5. Risks

- Import-path churn is the main risk; mitigated by keeping `core` internally
  unchanged and verifying with pytest after each logical step.
- `mp` extra intentionally not in `.[dev]`; the one MP test skips without a key
  (unchanged from baseline).
</content>
