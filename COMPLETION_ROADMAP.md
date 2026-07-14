# Completion Roadmap: M1-M6 to 100%

This document provides actionable tasks to complete Milestones 1-6 from their current state to 100% completion.

---

## M3: Retrieval and Heuristics (85% → 100%)

**Current Gap:** Using offline chemistry proxies instead of computed thermodynamic data from Materials Project/DFT/MACE.

### Task 3.1: Materials Project Integration

**Goal:** Add convex hull analysis and competing phase detection.

**Implementation Steps:**

1. **Add Materials Project client** (`synthesis_planner/materials_project.py`)
   ```python
   class MaterialsProjectClient:
       def __init__(self, api_key: str):
           # Use mp-api or pymatgen.ext.matproj
       
       def get_hull_energy(self, formula: str) -> float | None:
           """Get energy above hull for target formula"""
       
       def get_competing_phases(self, elements: list[str]) -> list[dict]:
           """Get stable phases in chemical system"""
       
       def get_decomposition_energy(self, formula: str) -> float | None:
           """Get decomposition energy"""
   ```

2. **Extend ThermoAnalysisResult schema** (`schema.py`)
   ```python
   @dataclass(frozen=True)
   class ThermoAnalysisResult:
       score: float
       # ... existing fields ...
       hull_energy_ev_per_atom: float | None = None
       competing_phases: tuple[str, ...] = field(default_factory=tuple)
       decomposition_energy_ev_per_atom: float | None = None
       formation_energy_ev_per_atom: float | None = None
   ```

3. **Add thermodynamic scorer** (`chemistry.py`)
   ```python
   def compute_thermodynamic_features(
       state: PlanningState,
       mp_client: MaterialsProjectClient | None = None
   ) -> ThermoAnalysisResult:
       # Existing offline proxies
       offline_score = _compute_offline_proxies(state)
       
       # Add MP data if available
       if mp_client:
           hull_energy = mp_client.get_hull_energy(state.problem.target_formula)
           competing = mp_client.get_competing_phases(state.target_elements)
           # Combine with offline features
       
       return ThermoAnalysisResult(...)
   ```

4. **Update scoring.py to use thermodynamic features**
   ```python
   def evaluate_state(...):
       # Add hull energy penalty
       if thermo.hull_energy_ev_per_atom is not None:
           if thermo.hull_energy_ev_per_atom > 0.1:  # Unstable
               thermo_score *= 0.5
       
       # Add competing phase penalty
       if thermo.competing_phases:
           # Check if precursors form stable competing phases
           thermo_score *= _competing_phase_penalty(...)
   ```

5. **Add config support for MP API key**
   ```python
   # config.example.py
   CONFIG = {
       "materials_project": {
           "api_key": "your-mp-api-key-here",
           "enable": True
       }
   }
   ```

6. **Add CLI flag to enable/disable MP integration**
   ```bash
   python run_mcts.py plan --target BaTiO3 --use-materials-project
   ```

**Testing:**
- Add `tests/test_materials_project.py` with mocked API responses
- Test fallback behavior when MP unavailable
- Verify scoring changes with/without MP data

**Deliverable:** Materials Project integration complete, scoring incorporates hull energies and competing phases.

**Estimated Effort:** 2-3 days

---

### Task 3.2: MACE Integration (Optional)

**Goal:** Fast structure relaxation and formation energy estimates using MACE foundation model.

**Implementation Steps:**

1. **Add MACE client** (`synthesis_planner/mace_client.py`)
   ```python
   class MACEClient:
       def __init__(self, model_path: str):
           # Load MACE model
       
       def relax_structure(self, formula: str, structure) -> dict:
           """Relax structure and return energy"""
       
       def estimate_formation_energy(self, formula: str) -> float | None:
           """Fast formation energy estimate"""
   ```

2. **Integrate with thermodynamic scoring**
   ```python
   def compute_thermodynamic_features(..., mace_client=None):
       if mace_client and not mp_client:
           # Use MACE as fast surrogate
           formation_energy = mace_client.estimate_formation_energy(...)
   ```

**Estimated Effort:** 3-5 days (depends on MACE model availability)

---

### Task 3.3: Reaction Driving Force Calculator

**Goal:** Compute ΔG or ΔH for precursor → target reactions.

**Implementation Steps:**

1. **Add reaction energetics** (`chemistry.py`)
   ```python
   def compute_reaction_driving_force(
       balance: ReactionBalanceResult,
       mp_client: MaterialsProjectClient
   ) -> float | None:
       """
       Compute ΔH_rxn = H_products - H_reactants
       Using formation energies from MP
       """
       target_energy = mp_client.get_formation_energy(target_formula)
       precursor_energies = [
           mp_client.get_formation_energy(p.formula) 
           for p in precursors
       ]
       # Include byproducts
       return compute_delta_H(...)
   ```

2. **Add to ThermoAnalysisResult**
   ```python
   @dataclass(frozen=True)
   class ThermoAnalysisResult:
       # ...
       reaction_driving_force_ev: float | None = None
       is_exothermic: bool | None = None
   ```

3. **Use in scoring**
   ```python
   if thermo.reaction_driving_force_ev is not None:
       if thermo.is_exothermic:
           thermo_score += 0.15  # Favorable
       else:
           thermo_score -= 0.1   # Endothermic may need higher T
   ```

**Estimated Effort:** 2 days

---

## M5: LLM Judge (70% → 100%)

**Current Gaps:** Calibration report, uncertainty quantification, partial-state judging.

### Task 5.1: Judge Calibration Report

**Goal:** Systematic report showing judge score correlation with recipe quality.

**Implementation Steps:**

1. **Add calibration module** (`synthesis_planner/judge_calibration.py`)
   ```python
   @dataclass(frozen=True)
   class CalibrationResult:
       judge_name: str
       n_samples: int
       correlation_with_validity: float
       correlation_with_precursor_match: float
       correlation_with_operation_similarity: float
       score_distribution: dict[str, float]
       high_score_precision: float  # What % of high-scored routes are valid?
       low_score_recall: float      # What % of invalid routes scored low?
       
   def calibrate_judge(
       judge: BaseJudge,
       test_routes: list[RouteRecord],
       split_type: str
   ) -> CalibrationResult:
       """
       Run judge on held-out routes and compute correlations
       between judge scores and ground-truth quality metrics
       """
       results = []
       for route in test_routes:
           # Convert RouteRecord to PlanningState
           state = _route_to_state(route)
           analogs = retrieval.retrieve(route.target_formula)
           hard_checks = check_hard_constraints(state)
           
           # Get judge score
           judge_result = judge.evaluate(state, analogs, hard_checks)
           
           # Compute ground-truth quality
           validity = hard_checks.valid
           precursor_match = _check_precursor_match(state, route)
           
           results.append({
               'judge_score': judge_result.score,
               'validity': validity,
               'precursor_match': precursor_match,
               ...
           })
       
       # Compute correlations
       from scipy.stats import pearsonr, spearmanr
       validity_corr = spearmanr([r['judge_score'] for r in results], 
                                  [r['validity'] for r in results])
       ...
       
       return CalibrationResult(...)
   ```

2. **Add CLI command**
   ```bash
   python run_mcts.py calibrate-judge \
       --judge deterministic \
       --split-type chemical_system \
       --output calibration_report.json
   ```

3. **Generate calibration plots** (`synthesis_planner/judge_calibration.py`)
   ```python
   def plot_calibration(result: CalibrationResult, output_path: str):
       import matplotlib.pyplot as plt
       
       # Score distribution histogram
       # Correlation scatter plots
       # Precision-recall curves
       # Save to output_path
   ```

4. **Add calibration to benchmark suite**
   ```python
   def run_benchmark_with_calibration(...):
       # Run normal benchmark
       summary = evaluate_split(...)
       
       # Add calibration
       calibration = calibrate_judge(judge, test_routes, split_type)
       
       return {
           'benchmark': summary.to_dict(),
           'calibration': asdict(calibration)
       }
   ```

**Testing:**
- Add `tests/test_judge_calibration.py`
- Test correlation computation with synthetic data
- Verify calibration report generation

**Deliverable:** Calibration report showing judge score correlations, precision/recall, score distributions.

**Estimated Effort:** 2-3 days

---

### Task 5.2: Uncertainty Quantification via Ensemble

**Goal:** Track judge uncertainty through disagreement across prompt variants or models.

**Implementation Steps:**

1. **Add ensemble judge** (`judge.py`)
   ```python
   class EnsembleJudge(BaseJudge):
       name = "ensemble"
       
       def __init__(self, config: dict):
           self.judges = [
               OpenAICompatibleStructuredJudge({
                   **config,
                   'prompt_variant': 'conservative'
               }),
               OpenAICompatibleStructuredJudge({
                   **config,
                   'prompt_variant': 'optimistic'
               }),
               OpenAICompatibleStructuredJudge({
                   **config,
                   'prompt_variant': 'skeptical'
               }),
           ]
       
       def evaluate(self, state, analogs, hard_checks) -> JudgeResult:
           results = [j.evaluate(state, analogs, hard_checks) for j in self.judges]
           
           # Compute ensemble statistics
           mean_score = mean([r.score for r in results])
           std_score = std([r.score for r in results])
           disagreement = max(r.score for r in results) - min(r.score for r in results)
           
           # Merge notes and flags
           all_notes = [note for r in results for note in r.notes]
           all_flags = [flag for r in results for flag in r.flags]
           
           return JudgeResult(
               score=mean_score,
               notes=tuple(set(all_notes)),
               flags=tuple(set(all_flags)),
               evidence_dois=results[0].evidence_dois,
               rubric_scores={
                   k: mean([r.rubric_scores.get(k, 0) for r in results])
                   for k in results[0].rubric_scores
               },
               uncertainty=disagreement  # Use disagreement as uncertainty proxy
           )
   ```

2. **Add prompt variants** (`judge.py`)
   ```python
   PROMPT_VARIANTS = {
       'conservative': """
           You are a skeptical materials chemist. Identify potential failure modes.
           Penalize unsupported novelty heavily. Default to uncertainty if evidence is weak.
       """,
       'optimistic': """
           You are an experienced synthesis chemist. Recognize when routes are plausible
           even if not exact literature matches. Give credit for reasonable adaptations.
       """,
       'skeptical': """
           You are an adversarial reviewer. Your goal is to find flaws in the proposed route.
           What could go wrong? What steps are missing? What assumptions are questionable?
       """
   }
   ```

3. **Update config to enable ensemble**
   ```python
   CONFIG = {
       "judge": {
           "name": "ensemble",
           "model": "gpt-4o-mini",
           "api_key": "...",
           "num_variants": 3
       }
   }
   ```

4. **Report uncertainty in results**
   ```python
   # In PlannedRoute output
   {
       "judge": {
           "score": 0.75,
           "uncertainty": 0.15,  # High disagreement = high uncertainty
           "notes": [...],
           ...
       }
   }
   ```

**Testing:**
- Add `tests/test_ensemble_judge.py`
- Verify disagreement computation
- Test with mocked responses showing high/low disagreement

**Deliverable:** Ensemble judge with uncertainty quantification via disagreement.

**Estimated Effort:** 2 days

---

### Task 5.3: Partial-State Judging

**Goal:** Evaluate incomplete routes to flag missing components early.

**Implementation Steps:**

1. **Add partial-state evaluation** (`judge.py`)
   ```python
   class BaseJudge:
       # Existing: evaluate(state, analogs, hard_checks) for terminal states
       
       def evaluate_partial(
           self, 
           state: PlanningState, 
           analogs: list[tuple[float, RouteRecord]]
       ) -> JudgeResult:
           """Evaluate incomplete route and flag likely missing steps"""
           
           if state.stage == "precursors":
               # After precursor selection, check coverage
               return self._check_precursor_completeness(state, analogs)
           
           elif state.stage == "preparation":
               # After prep, check if operations are sufficient
               return self._check_preparation_completeness(state, analogs)
           
           # ... other stages
   ```

2. **Add stage-specific checks** (`judge.py`)
   ```python
   def _check_precursor_completeness(self, state, analogs) -> JudgeResult:
       flags = []
       notes = []
       
       # Check element coverage
       if not _has_all_elements(state):
           flags.append("missing_element_source")
           notes.append("Precursor set does not cover all target elements")
       
       # Check for common missing components
       if _needs_oxidant(state) and not _has_oxidant(state.precursors):
           flags.append("missing_oxidant")
           notes.append("Target oxidation states suggest need for oxidizing agent")
       
       if _needs_reductant(state) and not _has_reductant(state.precursors):
           flags.append("missing_reductant")
           notes.append("Route may need reducing agent or atmosphere")
       
       return JudgeResult(score=0.5, notes=tuple(notes), flags=tuple(flags), ...)
   ```

3. **Integrate with MCTS expansion**
   ```python
   # In mcts.py
   def _expand(self, node, analogs, candidate_precursor_sets):
       # Before full expansion, optionally check partial state
       if self.evaluation_config.use_partial_judge and node.state.stage in CHECKABLE_STAGES:
           partial_result = judge.evaluate_partial(node.state, analogs)
           
           # If serious flags, penalize this branch
           if any(flag in BLOCKING_FLAGS for flag in partial_result.flags):
               node.value_penalty = 0.3
   ```

4. **Add config flag**
   ```python
   @dataclass(frozen=True)
   class EvaluationConfig:
       # ...
       use_partial_judge: bool = False  # Expensive, off by default
   ```

**Testing:**
- Add `tests/test_partial_judge.py`
- Test flag detection at each stage
- Verify partial evaluation doesn't break MCTS

**Deliverable:** Partial-state judging that flags missing components early in search.

**Estimated Effort:** 2-3 days

---

## M6: Retrospective Evaluation (60% → 100%)

**Current Gaps:** Failure-mode taxonomy, diversity-adjusted metrics, operation sequence alignment, analog support score.

### Task 6.1: Failure-Mode Taxonomy

**Goal:** Categorize benchmark errors by material family, synthesis modality, and failure type.

**Implementation Steps:**

1. **Define failure taxonomy** (`benchmark.py`)
   ```python
   @dataclass(frozen=True)
   class FailureMode:
       category: str  # precursor, operation, condition, validity, modality
       subcategory: str  # e.g., "volatile_element", "missing_regrind", "temp_too_low"
       description: str
       target_formula: str
       target_class: str
       modality: str
   
   FAILURE_CATEGORIES = {
       'precursor': [
           'wrong_precursor_class',
           'volatile_element_unhandled',
           'missing_element_source',
           'redox_mismatch'
       ],
       'operation': [
           'missing_mixing',
           'missing_regrind',
           'missing_wash_dry',
           'insufficient_heating'
       ],
       'condition': [
           'temperature_too_low',
           'temperature_too_high',
           'wrong_atmosphere',
           'insufficient_dwell_time'
       ],
       'validity': [
           'stoichiometry_imbalance',
           'element_coverage_failure',
           'redox_incompatibility'
       ]
   }
   ```

2. **Add failure analysis** (`benchmark.py`)
   ```python
   def analyze_failures(
       cases: list[BenchmarkCaseResult],
       test_routes: list[RouteRecord],
       planned_routes: dict[str, list[PlannedRoute]]
   ) -> list[FailureMode]:
       """Analyze failed cases and categorize by failure mode"""
       
       failures = []
       for case in cases:
           if not case.precursor_exact_match:
               # Analyze why precursors don't match
               failure_mode = _diagnose_precursor_failure(
                   case, test_routes, planned_routes
               )
               failures.append(failure_mode)
           
           if not case.top1_valid:
               # Analyze validity failure
               failure_mode = _diagnose_validity_failure(...)
               failures.append(failure_mode)
           
           if case.operation_similarity < 0.5:
               # Analyze operation mismatch
               failure_mode = _diagnose_operation_failure(...)
               failures.append(failure_mode)
       
       return failures
   ```

3. **Add failure diagnosis helpers** (`benchmark.py`)
   ```python
   def _diagnose_precursor_failure(case, test_routes, planned_routes):
       ground_truth = _find_route(test_routes, case.route_id)
       predicted = planned_routes[case.target_formula][0]
       
       # Compare precursor classes
       gt_classes = [p.class_name for p in ground_truth.precursors]
       pred_classes = [p.class_name for p in predicted.precursors]
       
       if 'carbonate' in gt_classes and 'oxide' in pred_classes:
           return FailureMode(
               category='precursor',
               subcategory='wrong_precursor_class',
               description='Used oxide instead of carbonate decomposition route',
               target_formula=case.target_formula,
               target_class=_classify(ground_truth.target_formula),
               modality=ground_truth.modality
           )
       
       # Check for volatile elements
       if any(el in VOLATILE_ELEMENTS for el in ground_truth.target_elements):
           if not _has_excess_volatile(predicted):
               return FailureMode(
                   category='precursor',
                   subcategory='volatile_element_unhandled',
                   description='Volatile element without excess precursor',
                   ...
               )
       
       # ... more diagnosis logic
   ```

4. **Generate taxonomy report** (`benchmark.py`)
   ```python
   def generate_taxonomy_report(failures: list[FailureMode]) -> dict:
       """Aggregate failures by category, material family, modality"""
       
       report = {
           'by_category': Counter(f.category for f in failures),
           'by_subcategory': Counter(f.subcategory for f in failures),
           'by_material_class': Counter(f.target_class for f in failures),
           'by_modality': Counter(f.modality for f in failures),
           'examples': defaultdict(list)
       }
       
       # Add representative examples for each subcategory
       for failure in failures:
           if len(report['examples'][failure.subcategory]) < 3:
               report['examples'][failure.subcategory].append({
                   'target': failure.target_formula,
                   'description': failure.description
               })
       
       return report
   ```

5. **Add to benchmark output**
   ```python
   @dataclass(frozen=True)
   class BenchmarkSummary:
       # ... existing fields ...
       failure_taxonomy: dict = field(default_factory=dict)
   
   def evaluate_split(...) -> BenchmarkSummary:
       # ... existing evaluation ...
       
       # Add failure analysis
       failures = analyze_failures(cases, test_routes, planned_routes)
       taxonomy = generate_taxonomy_report(failures)
       
       return BenchmarkSummary(..., failure_taxonomy=taxonomy)
   ```

**Testing:**
- Add `tests/test_failure_taxonomy.py`
- Create synthetic failure cases
- Verify categorization logic

**Deliverable:** Failure taxonomy report showing error patterns by material family and failure type.

**Estimated Effort:** 3-4 days

---

### Task 6.2: Diversity-Adjusted Route Quality

**Goal:** Penalize near-duplicate routes in top-k portfolio.

**Implementation Steps:**

1. **Add route similarity metric** (`planner.py`)
   ```python
   def route_similarity(route1: PlannedRoute, route2: PlannedRoute) -> float:
       """Compute similarity between two routes [0, 1]"""
       
       # Precursor similarity (Jaccard on formulas)
       precursor_sim = _jaccard_similarity(
           set(p.formula for p in route1.precursors),
           set(p.formula for p in route2.precursors)
       )
       
       # Operation similarity (sequence alignment)
       operation_sim = _sequence_similarity(
           [op.verb for op in route1.operations],
           [op.verb for op in route2.operations]
       )
       
       # Condition similarity
       condition_sim = _condition_similarity(
           route1.operations, route2.operations
       )
       
       # Weighted average
       return (0.4 * precursor_sim + 
               0.4 * operation_sim + 
               0.2 * condition_sim)
   ```

2. **Add diversity penalty to portfolio selection** (`planner.py`)
   ```python
   def _select_portfolio(
       terminal_routes: list[PlannedRoute], 
       top_k: int,
       diversity_threshold: float = 0.7  # Routes >70% similar penalized
   ) -> list[PlannedRoute]:
       """Select top-k diverse routes"""
       
       # Sort by score
       sorted_routes = sorted(
           terminal_routes, 
           key=lambda r: r.score.total, 
           reverse=True
       )
       
       # Greedily select diverse routes
       portfolio = []
       for route in sorted_routes:
           if len(portfolio) >= top_k:
               break
           
           # Check diversity vs existing portfolio
           max_similarity = max(
               (route_similarity(route, p) for p in portfolio),
               default=0.0
           )
           
           if max_similarity < diversity_threshold:
               portfolio.append(route)
           else:
               # Near-duplicate, skip unless very high score
               if route.score.total > portfolio[0].score.total * 1.1:
                   portfolio.append(route)
       
       return portfolio[:top_k]
   ```

3. **Add diversity metric to benchmarks** (`benchmark.py`)
   ```python
   @dataclass(frozen=True)
   class BenchmarkSummary:
       # ...
       mean_portfolio_diversity: float = 0.0
       
   def compute_portfolio_diversity(routes: list[PlannedRoute]) -> float:
       """Average pairwise dissimilarity in portfolio"""
       if len(routes) < 2:
           return 0.0
       
       similarities = [
           route_similarity(r1, r2)
           for i, r1 in enumerate(routes)
           for r2 in routes[i+1:]
       ]
       
       return 1.0 - mean(similarities)  # Convert to diversity
   ```

**Testing:**
- Add `tests/test_diversity.py`
- Test route similarity computation
- Verify diversity selection vs greedy top-k

**Deliverable:** Diversity-adjusted portfolio selection and diversity metrics in benchmarks.

**Estimated Effort:** 2 days

---

### Task 6.3: Operation Sequence Alignment

**Goal:** Replace basic operation overlap with sequence alignment (edit distance).

**Implementation Steps:**

1. **Add sequence alignment** (`benchmark.py`)
   ```python
   def operation_sequence_similarity(
       predicted_ops: list[str],
       ground_truth_ops: list[str]
   ) -> float:
       """
       Compute normalized edit distance between operation sequences
       Returns [0, 1] where 1 is perfect match
       """
       from difflib import SequenceMatcher
       
       # Normalize operation names (map to controlled vocab)
       pred_normalized = [_normalize_op(op) for op in predicted_ops]
       gt_normalized = [_normalize_op(op) for op in ground_truth_ops]
       
       # Compute sequence similarity
       matcher = SequenceMatcher(None, pred_normalized, gt_normalized)
       return matcher.ratio()
   ```

2. **Add operation normalization** (`benchmark.py`)
   ```python
   OPERATION_SYNONYMS = {
       'calcine': ['calcine', 'heat', 'fire'],
       'sinter': ['sinter', 'anneal', 'fire'],
       'grind': ['grind', 'mill', 'ball_mill', 'crush'],
       'mix': ['mix', 'blend', 'combine'],
       'wash': ['wash', 'rinse'],
       'dry': ['dry', 'heat'],
       'cool': ['cool', 'quench', 'air_cool'],
   }
   
   def _normalize_op(op: str) -> str:
       """Map operation to canonical name"""
       op_lower = op.lower()
       for canonical, synonyms in OPERATION_SYNONYMS.items():
           if op_lower in synonyms:
               return canonical
       return op_lower
   ```

3. **Replace basic similarity in benchmarks**
   ```python
   # Old: Jaccard similarity
   # New: Sequence alignment
   
   def evaluate_split(...):
       # ...
       for case in test_cases:
           # ...
           op_similarity = operation_sequence_similarity(
               predicted_ops, ground_truth_ops
           )
   ```

**Testing:**
- Add `tests/test_operation_alignment.py`
- Test with known sequence pairs
- Verify normalization logic

**Deliverable:** Operation sequence alignment using edit distance.

**Estimated Effort:** 1 day

---

### Task 6.4: Analog Support Score

**Goal:** Report analog support as explicit benchmark metric.

**Implementation Steps:**

1. **Add analog support to benchmark results** (`benchmark.py`)
   ```python
   @dataclass(frozen=True)
   class BenchmarkCaseResult:
       # ... existing fields ...
       analog_support_score: float = 0.0
       closest_analog_formula: str = ""
       closest_analog_similarity: float = 0.0
   ```

2. **Compute analog support during evaluation**
   ```python
   def evaluate_split(...):
       # ...
       for test_route in test_routes:
           # Plan route
           planned = planner.plan(problem, ...)
           
           # Retrieve analogs for this target
           analogs = retrieval.retrieve(test_route.target_formula)
           
           # Compute analog support for top predicted route
           analog_support = _compute_analog_support(
               planned[0], analogs
           )
           
           cases.append(BenchmarkCaseResult(
               # ...
               analog_support_score=analog_support,
               closest_analog_formula=analogs[0][1].target_formula if analogs else "",
               closest_analog_similarity=analogs[0][0] if analogs else 0.0
           ))
   ```

3. **Define analog support metric**
   ```python
   def _compute_analog_support(
       route: PlannedRoute,
       analogs: list[tuple[float, RouteRecord]]
   ) -> float:
       """
       Measure how well the route is supported by retrieved analogs
       Based on precursor class match, operation overlap, condition proximity
       """
       if not analogs:
           return 0.0
       
       # Find most similar analog
       best_score = 0.0
       for similarity, analog in analogs[:5]:
           # Precursor class overlap
           precursor_match = _precursor_class_overlap(
               route.precursors, analog.precursors
           )
           
           # Operation overlap
           operation_match = _operation_overlap(
               route.operations, analog.operations
           )
           
           # Weight by retrieval similarity
           support = similarity * (0.5 * precursor_match + 0.5 * operation_match)
           best_score = max(best_score, support)
       
       return best_score
   ```

4. **Add to summary metrics**
   ```python
   @dataclass(frozen=True)
   class BenchmarkSummary:
       # ...
       mean_analog_support: float = 0.0
   
   # In evaluate_split
   summary = BenchmarkSummary(
       # ...
       mean_analog_support=mean([c.analog_support_score for c in cases])
   )
   ```

**Testing:**
- Add test cases with high/low analog support
- Verify metric computation

**Deliverable:** Analog support score as explicit benchmark metric.

**Estimated Effort:** 1 day

---

## Summary of Effort Estimates

| Task | Milestone | Estimated Effort |
|------|-----------|-----------------|
| **M3.1:** Materials Project integration | M3 → 100% | 2-3 days |
| **M3.2:** MACE integration (optional) | M3 → 100% | 3-5 days |
| **M3.3:** Reaction driving force | M3 → 100% | 2 days |
| **M5.1:** Judge calibration report | M5 → 100% | 2-3 days |
| **M5.2:** Uncertainty quantification | M5 → 100% | 2 days |
| **M5.3:** Partial-state judging | M5 → 100% | 2-3 days |
| **M6.1:** Failure-mode taxonomy | M6 → 100% | 3-4 days |
| **M6.2:** Diversity-adjusted quality | M6 → 100% | 2 days |
| **M6.3:** Operation sequence alignment | M6 → 100% | 1 day |
| **M6.4:** Analog support score | M6 → 100% | 1 day |

**Total Estimated Effort (excluding MACE):** 18-22 working days (~3-4 weeks)

---

## Recommended Implementation Order

### Phase 1: Quick Wins (Week 1)
1. ✅ Task 6.3: Operation sequence alignment (1 day)
2. ✅ Task 6.4: Analog support score (1 day)
3. ✅ Task 6.2: Diversity-adjusted quality (2 days)
4. ✅ Task 5.1: Judge calibration report (2-3 days)

**Deliverable:** M6 metrics complete, judge calibration in place.

### Phase 2: Judge Improvements (Week 2)
5. ✅ Task 5.2: Uncertainty quantification (2 days)
6. ✅ Task 5.3: Partial-state judging (2-3 days)
7. ✅ Task 6.1: Failure-mode taxonomy (3-4 days)

**Deliverable:** M5 and M6 complete to 100%.

### Phase 3: Thermodynamic Integration (Week 3-4)
8. ✅ Task 3.1: Materials Project integration (2-3 days)
9. ✅ Task 3.3: Reaction driving force (2 days)
10. ⚠️ Task 3.2: MACE integration (optional, 3-5 days)

**Deliverable:** M3 complete to 100% (M1-M6 all at 100%).

---

## Testing Strategy

For each task, add corresponding test file:

```python
# tests/test_materials_project.py
def test_hull_energy_integration():
    # Mock MP API response
    # Verify scoring changes with hull energy data

# tests/test_judge_calibration.py
def test_calibration_report_generation():
    # Create synthetic routes with known quality
    # Verify correlation computation

# tests/test_failure_taxonomy.py  
def test_precursor_failure_diagnosis():
    # Create mismatched routes
    # Verify correct failure category

# tests/test_diversity.py
def test_diversity_penalty():
    # Create duplicate routes
    # Verify portfolio diversity increases

# tests/test_operation_alignment.py
def test_sequence_similarity():
    # Test with known operation sequences
    # Verify edit distance computation
```

---

## Dependencies and Prerequisites

### External Dependencies
- **Materials Project API:** Requires API key (free for academic use)
  - Install: `pip install mp-api`
  - Register at: https://next-gen.materialsproject.org/api
  
- **MACE (optional):** Requires model checkpoint
  - Install: `pip install mace-torch`
  - Model: Download from MACE repository
  
- **Scipy:** For correlation computation
  - Install: `pip install scipy`
  
- **Matplotlib (optional):** For calibration plots
  - Install: `pip install matplotlib`

### Configuration Changes
Add to `config.example.py`:
```python
CONFIG = {
    "materials_project": {
        "api_key": "your-mp-api-key",
        "enable": True
    },
    "judge": {
        "name": "ensemble",  # or "deterministic"
        "enable_partial_eval": False,
        "uncertainty_method": "ensemble"  # or "none"
    },
    "benchmark": {
        "enable_taxonomy": True,
        "diversity_threshold": 0.7,
        "use_sequence_alignment": True
    }
}
```

---

## Validation Criteria

### M3 Complete (100%)
- [ ] Materials Project API integrated
- [ ] Hull energies used in scoring
- [ ] Competing phases detected
- [ ] Reaction driving forces computed
- [ ] Tests pass with mocked MP responses
- [ ] Fallback to offline proxies when MP unavailable

### M5 Complete (100%)
- [ ] Calibration report generated for all judge types
- [ ] Correlation with validity >0.6
- [ ] Uncertainty quantification via ensemble disagreement
- [ ] Partial-state judging flags missing components
- [ ] Tests cover all judge evaluation modes

### M6 Complete (100%)
- [ ] Failure taxonomy report generated
- [ ] Errors categorized by material family and modality
- [ ] Portfolio diversity >0.5 on average
- [ ] Operation alignment uses edit distance
- [ ] Analog support score reported in benchmarks
- [ ] All metrics documented in benchmark output

---

## Next Steps After M1-M6 Completion

Once M1-M6 are at 100%, the natural next milestones are:

**M7: Prospective Validation (0% → 100%)**
- Select 5-10 target materials
- Generate route portfolios
- Execute in lab with XRD characterization
- Incorporate outcomes and re-calibrate

**M8: Discovery Integration (0% → 100%)**
- Connect to MCTS + MACE + DFT discovery workflow
- Add synthesizability as discovery reward
- Test closed-loop optimization

See `PROJECT_STATUS.md` for detailed roadmap beyond M6.
