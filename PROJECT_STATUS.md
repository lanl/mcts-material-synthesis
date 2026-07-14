# Project Status: MCTS Materials Synthesis Planner

**Generated:** 2026-07-14  
**Proposal:** MCTS_Materials_Synthesis_Project_Proposal.docx

## Executive Summary

This repository has **successfully implemented the core foundation** (approximately **60-70% of Milestone 1-5 deliverables**) described in the proposal. The planner is functional, tested, and produces ranked synthesis routes for solid-state, hydrothermal, and precipitation modalities. However, several proposal-specified capabilities remain **partially implemented or missing**, particularly around:

1. **Thermodynamic integration** (Materials Project/DFT/MACE-based physics scoring)
2. **Advanced LLM judge calibration** and uncertainty quantification
3. **Comprehensive benchmark evaluation** with expert review protocols
4. **Prospective experimental validation**
5. **Integration with existing discovery workflow** (MCTS + MACE + DFT)

---

## Proposal Specific Aims - Implementation Status

### ✅ Aim 1: Define a machine-actionable synthesis grammar
**Status: COMPLETE (100%)**

**What's Done:**
- `grammar.py`: Modality-aware finite grammar implemented for solid-state, hydrothermal, and precipitation
- Staged action expansion: precursors → preparation → heating/reaction → finalization
- Hierarchical action space prevents combinatorial explosion
- Operation vocabulary: mix, grind, mill, pelletize, calcine, sinter, anneal, wash, dry, cool, quench, precipitate, hydrothermal hold
- Solution-specific actions: solvent setup, pH/concentration, autoclave conditions, washing/drying
- Grammar is expressive enough for common literature recipes while tractable for MCTS

**Evidence:**
- `synthesis_planner/grammar.py` (414 lines)
- `synthesis_planner/schema.py` defines complete route representation
- Test coverage: `tests/test_constraints.py`, integration in `tests/test_planner.py`

---

### ⚠️ Aim 2: Build route priors and value functions from synthesis datasets
**Status: PARTIAL (70%)**

**What's Done:**
- ✅ Literature-mined datasets downloaded and normalized:
  - Solid-state: `CederGroupHub/text-mined-synthesis_public` (19,488 entries)
  - Solution-based: `CederGroupHub/text-mined-solution-synthesis_public` (35,675 entries)
- ✅ `datasets.py`: Complete normalization into canonical `RouteRecord` schema
- ✅ `retrieval.py`: Analog retrieval by target chemistry, element overlap, composition similarity
- ✅ Statistical priors: Precursor class frequencies, operation sequence priors, condition ranges from literature
- ✅ Route comparison, similarity features, evidence snippets with DOI metadata
- ✅ Stoichiometric balancing with volatile species (CO2, H2O, NO2, O2, etc.)
- ✅ Redox analysis: Oxidation state inference, environment compatibility checks
- ✅ Thermodynamic **proxy features**: Gas release/uptake, byproduct count, decomposition alignment, redox-environment matching

**What's Missing:**
- ❌ **Materials Project-style phase stability calculations** (convex hull analysis)
- ❌ **DFT reaction energies** and competing phase analysis
- ❌ **MACE-based surrogate calculations** for structure relaxation or finite-temperature features
- ❌ Direct integration with external thermodynamic APIs (currently offline proxies only)
- ⚠️ Learned value model (currently uses heuristic scoring + LLM judge)

**Rationale for Current State:**
The proposal states: *"Thermodynamic data from Materials Project-style phase stability calculations, DFT, and MACE-based surrogate calculations will be used to provide complementary physics-based scores."*

Current implementation uses **offline chemistry proxies** instead:
- Balanced reaction analysis
- Redox/oxidation state checks
- Decomposition temperature heuristics
- Gas evolution/uptake estimation

This allows the planner to work **offline** and remain **testable without external API dependencies**, but sacrifices the **quantitative thermodynamic rigor** specified in the proposal.

**Evidence:**
- `synthesis_planner/chemistry.py` (569 lines): Balancing, redox, thermo proxies
- `synthesis_planner/retrieval.py` (114 lines): Analog retrieval and precursor priors
- `synthesis_planner/datasets.py` (309 lines): Normalization pipeline
- README.md limitations section explicitly notes: *"Thermodynamic scoring now uses offline chemistry proxies derived from balanced reactions and redox checks, but it still does not use tabulated formation energies or phase-diagram data."*

---

### ⚠️ Aim 3: Develop an MCTS planner guided by constraints, retrieval, and LLM judging
**Status: STRONG PARTIAL (75%)**

**What's Done:**
- ✅ `mcts.py`: PUCT-style MCTS with selection, expansion, rollout, backup
- ✅ Policy priors from retrieval and learned route statistics
- ✅ Hard validity checks: element coverage, stoichiometric balance, redox/atmosphere compatibility, modality consistency, lab constraints
- ✅ `constraints.py`: Comprehensive hard-check gating with blocking flags
- ✅ Route scoring pipeline combining:
  - Hard validity
  - Stoichiometry
  - Precursor plausibility
  - Thermodynamic proxies (not full DFT)
  - Retrieval support
  - Condition plausibility
  - Cost/hazard/complexity penalties
- ✅ **Retrieval-grounded LLM judge** with structured output:
  - `DeterministicJudge`: Offline rubric-based baseline
  - `OpenAICompatibleStructuredJudge`: Model-backed structured evaluation
  - Judge receives retrieved analogs, hard-check results, and structured prompts
  - Rubric scores: precursor plausibility, condition compatibility, operation completeness, literature analogy, practicality
  - Judge outputs: score, notes, flags, evidence DOIs, rubric scores, uncertainty
- ✅ Judge is pluggable and ablation-friendly
- ✅ Top-k diverse portfolio selection

**What's Missing:**
- ⚠️ **Judge calibration against held-out recipes or expert ratings**: Calibration code exists in test fixtures but no systematic calibration report
- ⚠️ **Judge uncertainty quantification**: Schema supports `uncertainty` field, but no ensemble/disagreement tracking across model instances or prompt variants
- ⚠️ **Partial-state LLM judging**: Proposal suggests judge can evaluate incomplete routes (e.g., flag missing oxidant after precursor selection). Current implementation evaluates terminal routes only.
- ⚠️ **Active judge guardrails**: Prompt instructs evidence grounding, but no automated checks for unsupported novelty penalties

**Evidence:**
- `synthesis_planner/mcts.py` (96 lines): Compact PUCT search
- `synthesis_planner/scoring.py` (143 lines): Weighted scoring pipeline
- `synthesis_planner/judge.py` (420 lines): Pluggable judge interface with deterministic and OpenAI-compatible judges
- `synthesis_planner/constraints.py` (141 lines): Hard validity checks
- Test coverage: `tests/test_judge.py` (179 lines), `tests/test_scoring.py`, `tests/test_constraints.py`

---

### ⚠️ Aim 4: Establish retrospective, expert, and prospective evaluation protocols
**Status: PARTIAL (50%)**

**What's Done:**
- ✅ `benchmark.py`: Split generation, baseline methods, retrospective evaluation framework
- ✅ **Split types implemented:**
  - `random`
  - `target_formula` (target-disjoint)
  - `chemical_system` (element-combination disjoint)
  - `material_family` (structural prototype disjoint)
  - `publication_year` (time-based split)
- ✅ **Baseline methods:**
  - `mcts` (full planner)
  - `nearest_neighbor` (retrieval-only)
  - `frequency_prior` (statistical prior baseline)
  - `mcts_no_retrieval` (ablation)
  - `mcts_no_judge` (ablation)
  - `mcts_no_hard_checks` (ablation)
- ✅ **Retrospective metrics:**
  - Precursor exact match @ top-1
  - Precursor class match @ top-1
  - Top-1 validity rate
  - Operation sequence similarity
  - Temperature error
- ✅ CLI commands: `make-splits`, `benchmark --method suite`
- ✅ Benchmark results written to `benchmark_results/`

**What's Missing:**
- ❌ **Expert plausibility ratings**: No blind expert scoring of generated vs. baseline routes
- ❌ **Diversity-adjusted route quality**: Metrics don't penalize near-duplicate routes in top-k
- ❌ **Operation F1 or edit similarity**: Current metric is basic operation overlap, not sequence alignment
- ❌ **Condition tolerance scoring**: Temperature error computed, but no pH/solvent/atmosphere tolerance ranges
- ❌ **Analog support score**: Not separately reported in benchmark results
- ❌ **Prospective experimental validation**: No validation against actual lab experiments
- ❌ **Failure-mode taxonomy**: Proposal suggests analyzing errors by material family, synthesis modality, but no structured error categorization
- ⚠️ **Calibration report**: Benchmarks run, but no systematic report of judge/model calibration vs. held-out recipes

**Evidence:**
- `synthesis_planner/benchmark.py` (328 lines)
- CLI smoke tests in README.md
- `tests/test_benchmark.py` (46 lines)
- Benchmark results directory exists with output JSON files

**Proposal Quote:**
> "Retrospective evaluation will emphasize top-k recovery, chemical validity, operation sequence similarity, and condition accuracy. Prospective evaluation will emphasize whether ranked routes produce the target phase or a meaningful partial success under realistic laboratory constraints."

**Gap:** Prospective validation is entirely absent.

---

## Milestone Progress

| Milestone | Proposal Deliverable | Implementation Status | Completion |
|-----------|---------------------|----------------------|------------|
| **M1: Route schema and grammar** | Validated schema; parser for dataset records; grammar-based route generator | ✅ Complete | 100% |
| **M2: Dataset normalization** | Normalized route database; target-disjoint and chemistry-disjoint benchmark splits | ✅ Complete | 100% |
| **M3: Retrieval and heuristics** | Retrieval engine; baseline route generator; interpretable route scoring features | ✅ Complete (offline proxies instead of DFT) | 85% |
| **M4: MCTS planner** | End-to-end route planner producing ranked synthesis routes | ✅ Complete | 100% |
| **M5: LLM judge** | LLM scoring module; judge calibration report; ablation-ready interface | ⚠️ Partial: Module done, calibration report missing | 70% |
| **M6: Retrospective evaluation** | Benchmark tables; ablations; failure-mode taxonomy | ⚠️ Partial: Tables done, taxonomy missing | 60% |
| **M7: Prospective validation** | Prospective synthesis report; updated route-value calibration | ❌ Not started | 0% |
| **M8: Integration with discovery** | Synthesizability-aware discovery demonstration | ❌ Not started | 0% |

---

## Component Implementation Matrix

| Component | Proposal Role | Status | Notes |
|-----------|--------------|--------|-------|
| **Route schema** | Canonical representation of routes | ✅ Done | `schema.py` with frozen dataclasses |
| **Synthesis grammar** | Defines allowable actions | ✅ Done | Modality-aware staged grammar |
| **Retrieval engine** | Finds analogous literature recipes | ✅ Done | Element overlap, composition similarity |
| **Physics-based scorers** | Hard checks + quantitative features | ⚠️ Partial | Offline proxies, no DFT/MACE |
| **LLM judge** | Evaluates route plausibility | ✅ Done | Deterministic + OpenAI-compatible structured judge |
| **MCTS controller** | Allocates search effort | ✅ Done | PUCT-style with rollouts |

---

## Data Resources - Implementation Status

| Dataset | Proposal Requirement | Status | Notes |
|---------|---------------------|--------|-------|
| **Solid-state synthesis** | Precursor/operation/condition priors, retrieval, retrospective benchmarking | ✅ Complete | 19,488 entries normalized to JSONL |
| **Solution-based synthesis** | Hydrothermal/precipitation support | ✅ Complete | 35,675 entries normalized to JSONL |
| **A-Lab outcomes** | Outcome-labeled data for calibration | ❌ Not integrated | Proposal mentions as design guidance only |
| **Materials Project** | Convex hull, competing phases, decomposition energies | ❌ Not integrated | Offline proxies used instead |
| **DFT reaction energies** | Reaction driving forces | ❌ Not integrated | |
| **MACE surrogate models** | Fast structure relaxation, finite-T features | ❌ Not integrated | |

---

## Code Quality Assessment

**Strengths:**
- ✅ **Well-structured modular design**: Clean separation of concerns (schema, grammar, retrieval, scoring, judge, MCTS)
- ✅ **Strong test coverage**: 1,000+ lines of tests across 10 test files (24% of codebase)
- ✅ **Immutable dataclasses**: All planning state is frozen, enabling safe tree search
- ✅ **Type hints**: Consistent use of modern Python type annotations
- ✅ **Documentation**: README, ARCHITECTURE.md, DATASETS.md, CLAUDE.md
- ✅ **CLI interface**: User-friendly commands for download, prepare, plan, benchmark
- ✅ **Ablation-friendly**: Scoring components are independently weighted and switchable

**Areas for Improvement:**
- ⚠️ **Limited docstrings**: Most functions lack docstrings (proposal complexity warrants more inline documentation)
- ⚠️ **No integration tests for full pipeline**: Tests focus on individual modules, not end-to-end planning
- ⚠️ **Hardcoded weights**: Scoring weights are literals in `scoring.py`, not configurable
- ⚠️ **No logging framework**: Uses print statements instead of structured logging

---

## Key Gaps vs. Proposal

### 1. **Thermodynamic Integration (High Priority)**

**Proposal:**
> "Thermodynamic data from Materials Project-style phase stability calculations, DFT, and MACE-based surrogate calculations will be used to provide complementary physics-based scores."

**Current State:**
- Uses offline chemistry proxies (balancing, redox, decomposition heuristics)
- No convex hull analysis
- No competing phase detection from computed data
- No reaction driving forces from DFT

**Impact:** Planner cannot distinguish thermodynamically favored routes from kinetically trapped or competing-phase-prone routes using quantitative physics.

**Recommended Action:**
- Add Materials Project API client for phase diagrams and hull energies
- Integrate MACE for fast structure relaxation and formation energy estimates
- Add competing phase analyzer using computed stability data
- Extend scoring pipeline with physics-based reaction driving forces

---

### 2. **Judge Calibration and Uncertainty (Medium Priority)**

**Proposal:**
> "The judge should be calibrated against held-out literature routes, expert ratings, and any available failed synthesis outcomes. The framework should also track judge uncertainty and disagreement across prompt variants or model instances."

**Current State:**
- Judge structure supports uncertainty field
- No systematic calibration report
- No ensemble/disagreement tracking
- No failed-outcome dataset integration

**Impact:** Judge scores may not correlate with actual synthesis success. Uncertainty estimates are not validated.

**Recommended Action:**
- Generate calibration report comparing judge scores to held-out recipe quality
- Implement ensemble judging (multiple prompt variants or models)
- Track disagreement as uncertainty proxy
- Integrate A-Lab-style failed outcomes if available

---

### 3. **Expert Evaluation and Prospective Validation (High Priority)**

**Proposal:**
> "The planner will be evaluated against held-out literature recipes, route-plausibility baselines, expert review, and prospective experimental validation where feasible."

**Current State:**
- Retrospective metrics implemented
- **No expert review protocol**
- **No prospective experimental validation**

**Impact:** Cannot validate whether high-scoring routes are actually synthesizable in practice.

**Recommended Action:**
- Design blind expert evaluation protocol (chemists rate generated vs. literature routes)
- Select small target set for prospective lab validation
- Define success criteria (XRD phase purity, Rietveld refinement)
- Incorporate outcomes into route schema and re-calibrate

---

### 4. **Advanced Evaluation Metrics (Medium Priority)**

**Proposal Metrics Not Implemented:**
- Diversity-adjusted route quality (penalize near-duplicates)
- Operation F1 or edit similarity (sequence alignment)
- Analog support score (separate metric)
- Failure-mode taxonomy by material family

**Recommended Action:**
- Add diversity penalty to portfolio selection
- Implement sequence alignment for operation similarity
- Add analog support as explicit metric
- Categorize benchmark errors by target class, modality, failure mode

---

### 5. **Integration with Discovery Workflow (Low Priority for Phase 1)**

**Proposal:**
> "Connect planner to existing MCTS + MACE + DFT material discovery workflow as a downstream filter or reward term."

**Status:** Not started (M8).

**Rationale:** Proposal intentionally separates synthesis planning from discovery as a cleaner initial contribution. This gap is expected and appropriate for the current phase.

---

## What Works Well Right Now

1. **End-to-end planning functional**: Can generate ranked routes for any target and modality
2. **Modality-aware grammar**: Solid-state, hydrothermal, and precipitation routes are chemically sensible
3. **Retrieval-grounded**: Routes are anchored to literature analogs, not hallucinated
4. **Hard constraints prevent nonsense**: Routes pass element coverage, stoichiometry, redox checks
5. **Ablation-ready**: Can easily test without judge, retrieval, or hard checks
6. **Benchmark framework**: Infrastructure exists to test generalization across splits
7. **Pluggable judge**: Easy to add new judge implementations
8. **Offline operation**: No external API dependencies for core functionality

---

## Recommended Next Steps (Priority Order)

### Phase 1: Strengthen Core Capabilities (2-3 months)

1. **Thermodynamic Integration**
   - Add Materials Project API client for hull energies and competing phases
   - Integrate MACE for fast surrogate energetics (if model available)
   - Extend scoring with quantitative reaction driving forces
   - Add competing phase analyzer

2. **Judge Calibration**
   - Generate systematic calibration report (judge scores vs. held-out recipe quality)
   - Implement ensemble judging with disagreement tracking
   - Add failed-outcome dataset if available (A-Lab style)

3. **Enhanced Metrics**
   - Add diversity penalty to portfolio selection
   - Implement operation sequence alignment (edit distance)
   - Add failure-mode taxonomy to benchmarks
   - Report analog support as explicit metric

### Phase 2: Expert Validation (2-3 months)

4. **Expert Review Protocol**
   - Design blind evaluation (rate generated vs. literature routes)
   - Recruit domain experts for route plausibility ratings
   - Analyze correlation between model scores and expert ratings

5. **Prospective Validation**
   - Select 5-10 targets across chemical families
   - Generate route portfolios under real lab constraints
   - Execute experiments with XRD characterization
   - Incorporate outcomes and re-calibrate

### Phase 3: Discovery Integration (3-6 months)

6. **Discovery Workflow Integration**
   - Connect to existing MCTS + MACE + DFT discovery pipeline
   - Add synthesizability as discovery reward term
   - Test closed-loop discovery-synthesis optimization
   - Publish integrated framework results

---

## Code Statistics

| Metric | Count |
|--------|-------|
| **Total source lines** | ~3,400 (synthesis_planner + tests) |
| **Main package lines** | ~3,000 (synthesis_planner/) |
| **Test lines** | ~1,000 (tests/) |
| **Test coverage** | ~24% of codebase |
| **Modules** | 14 (schema, grammar, mcts, judge, scoring, etc.) |
| **Test files** | 10 |
| **Judge implementations** | 3 (Base, Null, Deterministic, OpenAI-compatible) |

---

## Conclusion

The repository has built a **strong foundation** matching 60-70% of the proposal's Milestones 1-5. The core MCTS planner, grammar, retrieval, and LLM judge are **functional and well-tested**. However, three major gaps remain:

1. **No integration with computed thermodynamics** (DFT, MACE, Materials Project)
2. **No expert evaluation or prospective validation**
3. **Missing advanced calibration and uncertainty quantification**

The current system can **generate plausible routes** and is **ready for research use**, but **cannot claim the quantitative thermodynamic rigor or experimental validation** described in the proposal.

**Bottom line:** This is a **strong MVP** for Milestones 1-5, but requires thermodynamic integration and experimental validation to become the **complete synthesis planner** envisioned in the proposal.
