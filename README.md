# MCTS Materials Synthesis Planner

This repo is now a synthesis-planning project aligned with the proposal in [MCTS_Materials_Synthesis_Project_Proposal.docx](./MCTS_Materials_Synthesis_Project_Proposal.docx). The old crystal-search package and its study directories have been removed so the repository now reflects the new synthesis-first scope directly.

The current codebase focuses on the proposal's first practical milestone:

- public dataset download and normalization
- a canonical route schema for mined synthesis recipes
- modality-aware synthesis grammars for solid-state, hydrothermal, and precipitation planning
- retrieval of analogous literature routes
- route scoring with explicit stoichiometric balancing, redox-aware hard checks, thermodynamic proxy features, and a pluggable retrieval-grounded judge interface
- Monte Carlo Tree Search over partial synthesis routes
- retrospective split generation, baselines, and ablation-style benchmark evaluation

The initial planner is intentionally a strong solid-state baseline, not a claim that the full proposal is finished. The solution-based dataset is downloaded and normalized for future hydrothermal and precipitation support, but the executable planner currently targets `solid_state` routes.
The planner now supports executable `solid_state`, `hydrothermal`, and `precipitation` search, with a deterministic judge by default and benchmark support for nearest-neighbor and frequency-prior baselines.

## Public data

The repo now uses the public datasets referenced in the proposal:

- Solid-state dataset: `CederGroupHub/text-mined-synthesis_public`
  - file: `solid-state_dataset_20200713.json.xz`
- Solution dataset: `CederGroupHub/text-mined-solution-synthesis_public`
  - file: `solution-synthesis_dataset_2021-8-5.json.zip`

They are downloaded into `data/raw/` and normalized into JSONL route records in `data/processed/`.

## Install

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel pytest numpy pandas
.venv/bin/python -m pip install .
```

For development:

```bash
.venv/bin/python -m pip install '.[dev]'
```

## Local config

The CLI now looks for local configuration in `config.py` first, then `config.json`.

- `config.py` is gitignored and safe for local secrets such as API keys.
- `config.example.py` is the tracked template showing the expected shape.
- `config.example.json` remains available if you prefer JSON for non-secret defaults.

If you use the model-backed judge, keep the real key only in local `config.py` or an environment variable such as `OPENAI_API_KEY`.

## Workflow

Download the public datasets:

```bash
python run_mcts.py download-data
```

Normalize them into route records:

```bash
python run_mcts.py prepare-data
```

Plan routes for a target:

```bash
python run_mcts.py plan --target BaTiO3 --iterations 250 --top-k 5
```

Plan a hydrothermal route:

```bash
python run_mcts.py plan --target CoFe2O4 --modality hydrothermal
```

Plan a precipitation route:

```bash
python run_mcts.py plan --target TiO2 --modality precipitation
```

Run with ablation-style switches:

```bash
python run_mcts.py plan --target BaTiO3 --disable-retrieval
python run_mcts.py plan --target BaTiO3 --judge none --disable-judge
python run_mcts.py plan --target BaTiO3 --disable-hard-checks
```

Use a retrieval-grounded structured judge with an OpenAI-compatible endpoint:

```bash
python run_mcts.py plan \
  --target BaTiO3 \
  --judge openai_structured \
  --judge-model gpt-4o-mini
```

You can also override credentials or endpoint per run:

```bash
python run_mcts.py plan \
  --target BaTiO3 \
  --judge openai_structured \
  --judge-model gpt-4o-mini \
  --judge-api-key "$OPENAI_API_KEY" \
  --judge-base-url "https://api.openai.com/v1"
```

Generate benchmark splits:

```bash
python run_mcts.py make-splits --split-type target_formula
```

Run a small retrospective benchmark:

```bash
python run_mcts.py benchmark --split-type chemical_system --iterations 50 --rollout-count 3
```

Run baseline and ablation comparisons:

```bash
python run_mcts.py benchmark --method nearest_neighbor
python run_mcts.py benchmark --method frequency_prior
python run_mcts.py benchmark --method suite
```

The planner writes ranked routes to `planning_results/` and prints a short summary to the terminal.

## Documentation

- [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md): package layout, planning pipeline, and current design boundaries
- [docs/DATASETS.md](./docs/DATASETS.md): public dataset sources, local file layout, and normalization outputs

## What the planner does

Given a target such as `BaTiO3`, the current planner:

1. Loads normalized solid-state routes from the mined literature corpus.
2. Retrieves analogous targets by element overlap, target family, and rough composition similarity.
3. Builds precursor candidates from exact analog routes plus element-level precursor usage priors.
4. Expands a staged solid-state grammar:
   - precursor set
   - preparation steps
   - heating schedule
   - finalization
5. Uses modality-aware grammars for hydrothermal and precipitation when requested:
   - solvent setup
   - reaction/hold/precipitation step
   - washing, drying, and optional post-anneal
6. Applies hard validity checks for element coverage, modality consistency, lab constraints, temperature bounds, atmosphere compatibility, solvent requirements, exact precursor-to-target balancing with common volatile species, and redox/environment consistency.
7. Scores completed routes using:
   - element coverage and balanced reaction feasibility
   - hard validity
   - precursor plausibility
   - thermodynamic proxy features such as gas release, decomposition alignment, and redox support
   - retrieval support
   - condition plausibility
   - hazard and complexity penalties
   - retrieval-grounded judge notes, flags, rubric scores, and uncertainty
8. Uses MCTS to prioritize promising partial routes and returns a small diverse portfolio.

## Repo layout

- `docs/`
  - architecture and dataset notes for the rewritten planner
- `data/`
  - `raw/`: downloaded public corpora
  - `processed/`: normalized JSONL route records generated locally
- `synthesis_planner/`
  - `datasets.py`: dataset download, loading, normalization
  - `formula.py`: formula parsing and target-family heuristics
  - `chemistry.py`: stoichiometric balancing, oxidation/redox checks, and thermodynamic proxy features
  - `retrieval.py`: analog retrieval and precursor prior generation
  - `grammar.py`: staged solid-state grammar
  - `constraints.py`: modality-aware hard validity checks
  - `judge.py`: deterministic and model-backed structured retrieval-grounded judges
  - `scoring.py`: chemistry-aware route scoring plus judge integration
  - `mcts.py`: compact PUCT-style tree search
  - `planner.py`: high-level planning interface
  - `benchmark.py`: split generation, baselines, and evaluations
  - `cli.py`: `download-data`, `prepare-data`, `plan`, `make-splits`, `benchmark`
- `tests/`
  - focused on formula parsing, normalization, retrieval, scoring, planner behavior, and CLI parsing

## Current limitations

- Thermodynamic scoring now uses offline chemistry proxies derived from balanced reactions and redox checks, but it still does not use tabulated formation energies or phase-diagram data.
- The judge interface is pluggable, and now supports a structured OpenAI-compatible judge, but the default remains deterministic so the repo still works offline and remains testable.
- Route scoring is still a baseline heuristic layer rather than a calibrated literature-plus-physics model.
- The benchmark harness now includes baselines and simple ablations, but it still does not cover the full expert/prospective evaluation loop from the proposal.

## Verification

Unit tests:

```bash
.venv/bin/python -m pytest
```

CLI smoke-tested commands:

```bash
.venv/bin/python run_mcts.py make-splits --split-type target_formula
.venv/bin/python run_mcts.py benchmark --split-type random --test-fraction 0.0002 --iterations 8 --rollout-count 2
.venv/bin/python run_mcts.py benchmark --method suite --split-type random --test-fraction 0.0002 --iterations 6 --rollout-count 2
.venv/bin/python run_mcts.py plan --target CoFe2O4 --modality hydrothermal
.venv/bin/python run_mcts.py plan --target TiO2 --modality precipitation
```

The rewritten test suite covers the new synthesis-planning path rather than the legacy crystal-search code.
