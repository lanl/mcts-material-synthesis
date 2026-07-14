# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Monte Carlo Tree Search (MCTS) based synthesis planner for inorganic materials. The planner generates synthesis routes for target compounds like BaTiO3, using literature-mined recipes, chemistry-aware scoring, and modality-specific grammars (solid-state, hydrothermal, precipitation).

## Development Setup

**Install for development:**
```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel pytest numpy pandas
.venv/bin/python -m pip install '.[dev]'
```

**Run tests:**
```bash
.venv/bin/python -m pytest
```

**Run single test:**
```bash
.venv/bin/python -m pytest tests/test_cli.py::test_plan_subcommand_parses_target
```

## Core Commands

**Data preparation workflow:**
```bash
python run_mcts.py download-data    # Download public datasets to data/raw/
python run_mcts.py prepare-data     # Normalize to JSONL in data/processed/
```

**Planning routes:**
```bash
python run_mcts.py plan --target BaTiO3 --iterations 250 --top-k 5
python run_mcts.py plan --target CoFe2O4 --modality hydrothermal
python run_mcts.py plan --target TiO2 --modality precipitation
```

**Benchmark evaluation:**
```bash
python run_mcts.py make-splits --split-type target_formula
python run_mcts.py benchmark --split-type chemical_system --iterations 50 --rollout-count 3
python run_mcts.py benchmark --method nearest_neighbor
python run_mcts.py benchmark --method suite
```

## Architecture

### Core Planning Pipeline

1. **Data ingestion** (`datasets.py`): Downloads and normalizes public synthesis corpora (solid-state, solution) into `RouteRecord` JSONL format
2. **Retrieval** (`retrieval.py`): Finds analogous target recipes and builds precursor usage priors
3. **Grammar expansion** (`grammar.py`): Modality-aware action generation across stages:
   - `precursors`: Select precursor set from candidates
   - `preparation`: Mixing, grinding, ball milling, pelletizing
   - `heating`/`reaction`: Temperature/atmosphere schedules (modality-specific)
   - `finalize`: Cooling, washing, drying
4. **Hard constraints** (`constraints.py`): Element coverage, stoichiometry, redox/atmosphere compatibility, modality consistency
5. **Scoring** (`scoring.py`): Chemistry-aware route evaluation combining validity, stoichiometry, precursor plausibility, thermodynamic proxies, retrieval support, and judge scores
6. **Judge layer** (`judge.py`): Pluggable retrieval-grounded evaluation (deterministic by default, optional OpenAI-compatible structured judge)
7. **MCTS search** (`mcts.py`): PUCT-style tree search with rollouts and backpropagation
8. **Portfolio selection** (`planner.py`): Returns top-k diverse routes

### Key Modules

- `schema.py`: Core dataclasses (`RouteRecord`, `PlanningState`, `Action`, `ScoreBreakdown`, `JudgeResult`, etc.)
- `formula.py`: Inorganic formula parsing and target family inference
- `chemistry.py`: Stoichiometric balancing with volatile species (CO2, H2O, NO2, O2), redox analysis, thermodynamic proxy features
- `benchmark.py`: Split generation (target_formula, chemical_system, random), baseline methods (nearest_neighbor, frequency_prior), retrospective evaluation

### Modality-Aware Design

The planner supports three synthesis modalities with distinct grammar paths:

- **solid_state**: precursors → preparation (grind/mill) → heating → finalize (cool)
- **hydrothermal**: precursors → solvent setup → hydrothermal hold → wash/dry → optional anneal
- **precipitation**: precursors → solvent setup → precipitation → wash/dry → optional anneal

Each modality has:
- Dedicated expansion logic in `grammar.py` (`_expand_solution_state`, `_apply_solution_action`)
- Modality-specific hard checks in `constraints.py` (solvent requirements, temperature bounds, atmosphere rules)
- Modality-specific scoring in `judge.py` (decomposition windows, autoclave ranges, post-processing completeness)

## Configuration

The CLI loads configuration from `config.py` (gitignored) or `config.json`. Use `config.example.py` or `config.example.json` as templates.

**For OpenAI-compatible judge:**
```python
# config.py
CONFIG = {
    "judge": {
        "name": "openai_structured",
        "model": "gpt-4o-mini",
        "api_key": "your-key-here",
        "base_url": "https://api.openai.com/v1"
    }
}
```

CLI overrides:
```bash
python run_mcts.py plan --target BaTiO3 \
  --judge openai_structured \
  --judge-model gpt-4o-mini \
  --judge-api-key "$OPENAI_API_KEY" \
  --judge-api-style chat_completions
```

## Data Layout

```
data/
  raw/              # Downloaded archives (solid-state_dataset_20200713.json.xz, solution-synthesis_dataset_2021-8-5.json.zip)
  processed/        # Normalized JSONL route records by modality
planning_results/   # Generated synthesis plans (JSON)
benchmark_results/  # Evaluation outputs
```

## Testing Strategy

Tests focus on:
- Formula parsing and target family inference
- Dataset normalization and route schema compliance
- Retrieval analog matching and precursor prior generation
- Stoichiometric balancing and redox analysis
- Hard constraint validation (modality-aware)
- Scoring component integration
- CLI argument parsing and config loading
- MCTS node selection and backpropagation logic

When adding features:
- Add unit tests for new chemistry logic (`test_formula.py`, `test_chemistry.py`)
- Add integration tests for new CLI commands (`test_cli.py`)
- Test both deterministic and model-backed judge paths when relevant

## Code Patterns

### Immutable Schema

All planning state is frozen dataclasses. To modify a state, create a new instance:
```python
new_state = PlanningState(
    problem=state.problem,
    target_elements=state.target_elements,
    target_class=state.target_class,
    stage="heating",  # Changed stage
    precursors=state.precursors,
    solvents=state.solvents,
    operations=state.operations + new_ops,  # Append operations
    evidence_dois=state.evidence_dois,
    analog_targets=state.analog_targets,
)
```

### Scoring Pipeline

Scoring is decomposed into independent components summed in `scoring.py`:
```python
score = (
    validity_score * VALIDITY_WEIGHT +
    stoich_score * STOICH_WEIGHT +
    precursor_score * PRECURSOR_WEIGHT +
    thermo_score * THERMO_WEIGHT +
    retrieval_score * RETRIEVAL_WEIGHT +
    condition_score * CONDITION_WEIGHT +
    judge_score * LLM_WEIGHT +
    cost_penalty +
    hazard_penalty +
    complexity_penalty
)
```

Each component returns a 0-1 score. Add new features by extending this pipeline.

### Grammar Extensions

To add a new action type:
1. Add stage enum to `schema.py` if needed
2. Extend `expand_state()` in `grammar.py` to return new actions
3. Extend `apply_action()` to handle the new action kind
4. Add hard checks in `constraints.py` if new validity rules apply
5. Add scoring considerations in `scoring.py` and `judge.py`

### Judge Interface

Judges implement `BaseJudge`:
```python
class CustomJudge(BaseJudge):
    name = "custom"
    
    def evaluate(self, state: PlanningState, analogs: list[tuple[float, RouteRecord]], hard_checks: HardCheckResult) -> JudgeResult:
        # Analyze state, return JudgeResult with score, notes, flags, rubric_scores
        return JudgeResult(score=0.8, notes=("example note",), flags=(), rubric_scores={"key": 0.8})
```

Register in `judge.py` `_get_judge()` factory.

## Current Limitations

- Thermodynamic scoring uses offline chemistry proxies (balanced reactions, redox checks) but does not query formation energies or phase diagrams
- Default judge is deterministic; model-backed judge requires API key and compatible endpoint
- Route scoring is baseline heuristics, not a calibrated literature-plus-physics model
- Benchmark harness provides retrospective splits and ablations but not full prospective expert evaluation

## Documentation

- `docs/ARCHITECTURE.md`: Detailed package layout and design boundaries
- `docs/DATASETS.md`: Public dataset sources and normalization workflow
- `MCTS_Materials_Synthesis_Project_Proposal.docx`: Original project proposal and research plan
