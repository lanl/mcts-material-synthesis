# Critical Evaluation: `mcts-material-synthesis` vs. the Project Proposal

*Read-only assessment. No code was modified. pytest at time of review: **89 passed, 1 skipped**
(`test_materials_project` skips without an MP API key). Repo state: branch `modularize`
has been merged into `main` (current branch `main`, commit `3c1a903`).*

---

## Executive Summary

The repository is a clean, well-structured, well-tested **scaffold** that implements the
*shape* of the proposal end-to-end: dataset ingestion/normalization, a canonical route
schema, modality-aware grammars (solid-state / hydrothermal / precipitation), retrieval of
literature analogs, hard chemical constraints with real stoichiometric balancing and redox
reasoning, a thermodynamic *proxy* layer (plus optional Materials Project hookup), a
pluggable judge (deterministic-by-default, with a real optional OpenAI-compatible judge), a
bespoke PUCT MCTS engine, retrospective split generation, baselines, and ablation switches.
The engineering quality is high and the module boundaries are sound.

**But the scientific core of the proposal is not yet demonstrated, and in places is
architecturally undercut.** Three findings dominate:

1. **No empirical results exist.** `data/` is empty (no raw, no processed, no filtered),
   and there are no `planning_results/` or `benchmark_results/`. Every test runs on a
   2-record synthetic fixture. Nothing in the repo shows the planner recovering real
   recipes, beating a baseline, or that the judge is calibrated. The proposal's entire
   evaluation plan (Aim 4) is unexecuted.

2. **The search space is so small and template-driven that MCTS contributes little.** The
   grammar has ~4 shallow stages with 2-3 templated actions each; precursors come entirely
   from retrieval/frequency priors. The pipeline is **retrieval-dominated template
   filling**, not genuine search over synthesis-procedure space. This directly threatens
   the proposal's headline novelty ("move MCTS into searching synthesis-procedure space").

3. **The "calibrated retrieval-grounded LLM-as-judge" is only half-built.** Retrieval
   grounding is genuinely wired into the judge context. But the default judge is a
   hand-written rule engine, the LLM judge is untested against a live endpoint, and the
   calibration routine correlates the judge against `hard_checks.valid` on
   literature-only (nearly all-valid) routes — a partly circular, negatives-free setup that
   cannot establish the calibration the proposal requires.

**Verdict on viability:** The framework is a credible *foundation* and roughly realizes the
proposal's plumbing. It does **not** yet realize the proposal's scientific thesis, and a
Nature-tier claim is far out of reach at the current state — not because of code quality,
but because zero validation has been run and the method as built may not beat nearest-neighbor
retrieval. This is fixable, but the work remaining is the *hard scientific* part, not more
scaffolding.

---

## 1. Viability vs. the Proposal (goal → status)

| Proposal element | Status | Evidence |
|---|---|---|
| Aim 1: machine-actionable synthesis grammar (solid-state) | **Implemented (shallow)** | `core/grammar.py` `expand_state`/`apply_action`: stages precursors→preparation→heating→finalize; heating templates single/staged/literature-multistep |
| Aim 1: hydrothermal + precipitation extension | **Implemented (shallow)** | `grammar._expand_solution_state` / `_apply_solution_action`, solvent/reaction/postprocess stages |
| Aim 2: route priors from datasets (precursor/operation/condition) | **Partial** | `data/retrieval.py` builds precursor & class usage priors by element; heating temps = median of analog temps. Operation-sequence priors are *not* learned (templates are hard-coded) |
| Aim 2: normalized common route schema | **Implemented (lossy)** | `core/schema.py RouteRecord`; but drops `amount`, `hydrate_state`, `role`, and `outcome` fields the proposal schema specifies |
| Aim 2: Materials Project / DFT / MACE physics scores | **Partial / stubbed** | `data/materials_project.py` real MP client (needs key, 1 test skipped); `chemistry._compute_reaction_driving_force` uses MP + hardcoded gas ΔH. **No DFT, no MACE** despite proposal |
| Aim 3: MCTS with PUCT selection/expansion/rollout/backup | **Implemented (but low-value, see §6)** | `core/mcts.py` PUCT; rollouts via `grammar.rollout_completion` |
| Aim 3: hard validity + soft scoring | **Implemented (heuristic)** | `core/constraints.py` (coverage, balance, redox, modality, lab limits); `core/scoring.py` weighted sum |
| Aim 3: retrieval-grounded LLM judge as *heuristic not oracle* | **Partial** | `core/judge.py`: real `OpenAICompatibleStructuredJudge` with retrieval context + JSON schema + fallback; default is `DeterministicJudge` (rules, not an LLM) |
| Aim 3: partial-state judging | **Implemented (unused in search)** | `DeterministicJudge.evaluate_partial`; **not called by MCTS** (`use_partial_judge` flag exists in `EvaluationConfig` but is never consumed) |
| Aim 4: retrospective splits (target/chemsys/family/pub-time) | **Implemented** | `benchmark.build_split` supports all four + random |
| Aim 4: baselines (NN, frequency, beam, LLM-only, ablations) | **Partial** | NN, frequency_prior, and 3 ablations present; **beam search and LLM-only planner absent**; no random baseline |
| Aim 4: retrospective metrics | **Partial** | precursor exact/class match, top-1 validity, operation similarity, temperature error; **no top-k recall, no condition-range accuracy, no novelty** |
| Aim 4: judge calibration | **Present but methodologically weak** | `judge_calibration.py` (see §6 circularity) |
| Aim 4: expert review + prospective experiments | **Absent** | Not started; acknowledged in README/ARCHITECTURE |
| Portfolio diversity output | **Implemented** | `planner._select_portfolio` greedy diversity over precursor/op/temp similarity |
| Integration with discovery (synthesizability reward) | **Absent (by design, deferred)** | Correctly out of scope for stage 1 |

---

## 2. What it IS doing vs. what it is NOT

### Genuinely doing (evidence-backed)
- **Real stoichiometric balancing.** `chemistry.balance_route` sets up a fractional linear
  system over framework elements, finds a nonnegative basic-feasible precursor combination
  (`_solve_basic_feasible`), then solves the volatile residual against common gaseous
  products/reactants (`_solve_volatile_residual`, CO2/H2O/NH3/NOx/O2/halogens). Uses exact
  `Fraction` arithmetic and produces a formatted balanced equation. This is a real,
  non-trivial chemistry engine (`test_chemistry.py`, `test_reaction_driving_force.py`).
- **Redox / atmosphere reasoning.** `chemistry.analyze_redox` infers focus-element charge
  from fixed oxidation-state tables and flags `missing_oxidant`/`missing_reductant`/
  atmosphere mismatches; these become blocking hard-check flags in `constraints.py`.
- **Hard-constraint gating** with explicit blocking flags: element coverage, precursor
  count, forbidden classes, modality-inconsistent operations, missing heat/mix/solvent,
  temperature bounds, atmosphere allow-lists, precipitation post-processing.
- **Retrieval** with Jaccard element overlap + composition-distance + exact-target and
  class bonuses (`retrieval.RetrievalIndex.retrieve`), and precursor candidate assembly
  from analogs and per-element frequency priors (`candidate_precursor_sets`).
- **A working PUCT MCTS** with backup and terminal-route accumulation, plus deterministic
  seeding.
- **A real optional model judge** (`OpenAICompatibleStructuredJudge`) that packages target,
  candidate route, hard-check output, and top-5 retrieved analogs (with paragraph excerpts,
  DOIs) into a JSON-schema-constrained request, with responses/chat-completions styles,
  JSON repair, and graceful fallback to the deterministic judge.
- **Benchmark harness** with disjoint splits (verified by `test_benchmark`), ablation
  method switches, operation-synonym normalization, and a failure taxonomy
  (`failure_taxonomy.py`).

### NOT doing (gaps, stubs, superficial)
- **No data and no results.** `data/processed/` is empty; no `data/raw/`, no
  `data/processed_filtered/`, no `planning_results/`, no `benchmark_results/`. Tests use the
  2-record fixture in `tests/conftest.py`. The reported "89 passed" verifies **plumbing on
  toy data**, not scientific behavior.
- **Operation/condition priors are not learned.** Heating schedules are hard-coded templates
  parameterized only by the *median* analog temperature/time (`grammar._heating_actions`).
  There is no learned operation-sequence model, no dwell-time or ramp modeling beyond a
  single midpoint, no atmosphere selection beyond "most common."
- **Thermodynamics is a proxy.** `analyze_thermodynamics` scores gas release, precursor-class
  decomposition matching, and redox support. Real energetics only appear if an MP key is
  configured; **no MACE, no DFT, no convex-hull computation** in-repo.
- **Partial-state judging is dead code in the search loop** — implemented but never invoked
  (`EvaluationConfig.use_partial_judge` is set but unused in `mcts.py`/`scoring.py`).
- **Schema loses quantitative recipe content.** `PrecursorRecord` has no amount / hydrate /
  role; `RouteRecord` has no `outcome`/`phase_purity`/`yield`. The proposal's schema (§6.5)
  explicitly includes these, and the outcome field is called "essential" for calibration.
- **Precursor classification is crude substring matching** (`datasets.classify_precursor`):
  e.g. `"oh" in lowered` will tag many non-hydroxide formulas, `"coo"` for acetate is
  fragile, sulfide = "S in parsed and O not in parsed." This feeds precursor-class metrics
  and scoring, so the noise propagates.
- **Formula parser rejects fractional/solid-solution notation** (see `filter_dataset.py`,
  which exists precisely to drop e.g. `(Y0.9Ca0.1)BaCo2O5.5`). Fine as a small exclusion,
  but doped/solid-solution targets — a proposal use case — are systematically dropped.
- **`material_family` split uses `target_class`** (oxide/sulfide/…), which is a coarse
  anion class, **not a structural prototype**. The proposal's "prototype/material-family
  split" (structural family) is therefore not really implemented.
- **`publication_year`** is scraped from a DOI regex (`_extract_publication_year`) — noisy
  and often absent, weakening the publication-time split.

---

## 3. What we need to implement (prioritized)

### Must-have for the scientific claim
1. **Run the real datasets end-to-end and produce actual results.** Download Kononova
   solid-state + Wang solution, normalize, generate all four splits, run MCTS vs. baselines,
   save metrics. *Effort: low-moderate (code exists). Risk: moderate — results may show MCTS
   ≈ nearest-neighbor, which is the real scientific test.*
2. **Demonstrate MCTS/grammar value over retrieval.** Add the missing **beam-search** and
   **LLM-only** baselines and a **random** baseline; report MCTS vs. NN vs. frequency with
   confidence intervals across seeds and splits. If MCTS does not beat NN, the central thesis
   needs rethinking. *Effort: moderate. Risk: high (it may not).*
3. **Deepen the action space so search matters.** Real per-family condition distributions
   (temperature ranges, dwell, ramp, atmosphere, regrind counts), excess/off-stoichiometry
   choices, alternative precursor chemistries — enough branching that PUCT explores a
   non-trivial space. *Effort: high. Risk: moderate.*
4. **Fix judge calibration to be non-circular and use negatives.** Calibrate against
   held-out *recovery* (does a high judge score predict recovering the true precursor set on
   target-disjoint splits?) and against **perturbed/corrupted negative routes** (wrong
   precursor, wrong atmosphere, missing step), not against `hard_checks.valid`. *Effort:
   moderate. Risk: moderate.*
5. **Add top-k recall and condition-range accuracy metrics.** Exact-match@1 is too strict
   (proposal says multiple routes are valid); add precursor-set recall@k, class recall@k,
   condition-in-range accuracy. *Effort: low.*
6. **Restore quantitative schema fields** (amounts, hydrate state, outcome) so calibration
   and thermodynamics can use them. *Effort: moderate (touches normalization + schema).*

### Nice-to-have / strengthening
7. Real thermodynamics via MACE surrogate or cached MP hulls for reaction driving force and
   competing-phase penalties (proposal explicitly wants MACE). *Effort: high. Risk: moderate.*
8. Wire in **partial-state judging** during search (it's already written). *Effort: low.*
9. Robust precursor classification (parse to composition + functional group, not substrings)
   and structural-prototype splits (via MP structure/spacegroup). *Effort: moderate.*
10. Ensemble-judge uncertainty actually used in ranking/portfolio, not just reported.
    *Effort: low.*

---

## 4. How to evaluate for a Nature(-level) paper

**Reality check first:** A Nature/Nature-family paper on synthesis planning would demand
(a) a clear, defensible novelty over existing text-mined-recipe recommenders and the A-Lab
recipe model, (b) large-scale retrospective results with proper leakage control, and
(c) ideally **prospective experimental validation**. The current repo has none of (b) or
(c), and (a) is at risk because the method is retrieval-dominated. A top-tier claim is **not
realistic now**; a solid *methods/benchmark* paper (e.g., npj Computational Materials,
Digital Discovery, Chem. Mater.) is a more honest near-term target, with Nature-tier reserved
for a version that adds genuine search value + prospective wins.

**What reviewers would demand, concretely:**

- **Baselines that isolate the contribution:** nearest-neighbor retrieval (present),
  frequency/template prior (present), **beam search over the same priors** (missing),
  **LLM-only planner** (missing), and the three ablations (present). The key figure is
  *MCTS − NN* on **target-disjoint** and **chemical-system** splits. If the gap is not
  clearly positive and significant, the paper has no thesis.
- **Metrics:** precursor-set recall@k and class recall@k (not just exact@1); operation-
  sequence similarity (present via `SequenceMatcher`); condition-range accuracy
  (temperature/dwell within analog range); route validity (present, but weak — see below);
  **novelty** (fraction of proposed routes not identical to any training route) paired with
  plausibility; calibration curves for the judge.
- **Statistical rigor:** multiple seeds, bootstrap CIs, per-split and per-material-class
  breakdowns, and explicit **leakage audits** (verify no test target/DOI/analog in train;
  the retrieval corpus must exclude test recipes — currently `evaluate_split` builds the
  `RetrievalIndex` from `train_routes`, which is correct, but the *precursor priors* are
  also train-only, good; this must be asserted and reported).
- **Ablations tied to failure modes** (the `failure_taxonomy.py` scaffold is the right idea):
  show removing thermo increases competing-phase errors, removing retrieval hurts condition
  accuracy, etc.
- **Judge validation** against expert ratings on a sampled set, and against corrupted-route
  negatives — to rebut the obvious "the LLM shares the generator's biases" critique.
- **Prospective validation** (proposal §10.4): even a handful of lab syntheses with XRD
  outcomes would be the difference between a methods paper and a high-impact one.

**Caveat on the current "validity" metric:** `top1_validity_rate` = `hard_checks.valid`.
Because routes are *constructed by a grammar that only emits balanceable, coverage-complete
recipes* and validity is then checked by the same balancing engine, this metric is close to
1.0 by construction and is **not discriminative**. Do not report it as a success rate;
reviewers will see the circularity immediately.

---

## 5. Dataset Composition (concrete findings)

- **Present locally: nothing usable.** `data/processed/` exists but is empty; there is no
  `data/raw/`, no `data/processed_filtered/`. `du` shows 26K (empty dir metadata). **No
  counts can be computed from local data — all numbers below are what the code expects to
  ingest, per proposal + normalization code, and are labeled inferred.**
- **Expected sources (verified in `data/datasets.py`):**
  - Solid-state: `CederGroupHub/text-mined-synthesis_public`,
    `solid-state_dataset_20200713.json.xz`, read via `payload["reactions"]`. *Proposal
    (inferred): ~19,488 synthesis entries / ~19,744 reactions (Kononova et al. 2019).*
  - Solution: `CederGroupHub/text-mined-solution-synthesis_public`,
    `solution-synthesis_dataset_2021-8-5.json.zip`. *Proposal (inferred): 35,675 procedures
    = 20,037 hydrothermal + 15,638 precipitation (Wang et al. 2022).* Modality is taken from
    each record's `type` field.
- **Normalization (verified):** each record → `RouteRecord{route_id, source_doi,
  publication_year(regex from DOI), modality, target_formula, target_elements,
  chemical_system, target_class, precursors[class via substring rules], solvents,
  operations[verb/temp/time/atmosphere], reaction_string, paragraph_excerpt,
  source_dataset}`. Temperatures normalized K→C, minutes→hours.
- **Known biases/gaps (partly acknowledged by proposal, partly introduced by code):**
  - Literature-mined → success-biased, **no true negatives / no outcome labels**; the schema
    doesn't even carry an `outcome` field, so A-Lab-style negative calibration is impossible
    as built.
  - Extraction noise inherited from the source datasets.
  - **Fractional/solid-solution formulas dropped** by the parser (`filter_dataset.py`
    exists solely to remove them; report claims ~0.08% but that is computed on the real data,
    not present here).
  - **Precursor-class labels are unreliable** (substring heuristics), which contaminates
    `precursor_class_match` metrics and precursor scoring.
  - Quantities/excess and hydrate state are discarded during normalization.
  - `target_class` is a coarse anion label; used as the "material family" split key, which is
    weaker than the proposal's intended structural-prototype split.
- **Provenance/licensing:** both corpora are the public CederGroupHub GitHub releases cited in
  the proposal (Kononova 2019; Wang 2022). Licensing terms were not audited here — **should
  be checked before redistribution** of processed data (processed files are gitignored, which
  is appropriate).

---

## 6. Is this a valid approach? (candid critique)

**The methodology is defensible in principle.** Framing inorganic synthesis as forward,
constrained sequential decision-making, gating with hard chemistry, ranking with soft
scores + retrieval + a grounded judge, and returning a diverse portfolio, is a reasonable and
honest design. The separation of synthesis planning from discovery (proposal's main
conceptual contribution) is sound. The hard-constraint engine (balancing + redox) is real and
is the strongest part of the codebase.

**However, several methodological risks are serious and currently unaddressed:**

1. **MCTS may be the wrong (or at least unproven) tool for this state space.** With ~4 stages,
   2-3 templated actions per non-precursor stage, and precursors drawn only from retrieval/
   frequency priors, the *entire* tree is tiny and the "search" is essentially enumerating a
   handful of template completions. `MonteCarloTreeSearch._simulate` even takes the **max**
   over rollouts rather than averaging, making it closer to optimistic enumeration than to
   value-averaged search. The proposal's novelty ("MCTS over synthesis-procedure space") is
   therefore not yet substantiated — the burden is to show MCTS beats beam search and NN on a
   space large enough to matter. *Mitigation: enlarge/branch the action space (real condition
   distributions, alternative chemistries, excess choices) and demonstrate a search-vs-greedy
   gap; otherwise reframe the contribution.*

2. **Retrieval leakage / retrieval domination.** Because both precursor candidates and analog
   scores derive from the corpus, and the retrieval score directly feeds `S_retrieval`, a
   naive random split will make everything look good via memorization. The harness *does*
   build retrieval from train-only routes (good), and provides target-disjoint / chemical-
   system splits (good). But no results exist to show performance holds under the hard splits,
   and there is no automated leakage assertion (e.g., ensuring no shared DOI/target between
   train retrieval index and test). *Mitigation: enforce and report leakage audits; headline
   the chemical-system split.*

3. **LLM-judge circularity and shared-bias risk.** The judge is fed retrieved analogs and
   hard-check output and asked to score the route. If the same retrieval/priors generate the
   route *and* justify it, the judge rewards self-consistency, not correctness. The proposal
   anticipates this ("unsupported novelty is a risk, not a virtue"), and the code does pass
   evidence + instructs evidence-only DOIs — but nothing validates that judge scores predict
   real-world synthesis success or even held-out recovery. *Mitigation: calibrate against
   corrupted-route negatives and expert labels; report precision/recall of the judge as a
   filter.*

4. **Circular calibration.** `judge_calibration.calibrate_judge` correlates the judge score
   with `hard_checks.valid`, but (a) the `DeterministicJudge` score is *derived from*
   `hard_checks.valid` (it subtracts 0.15 when invalid), and (b) held-out literature routes
   are almost all valid, so there are essentially no negatives. The reported correlation is
   therefore near-meaningless. *Mitigation: as in §3.4.*

5. **No ground-truth reactions / success labels.** The entire pipeline is validated against
   *reported* recipes, which are success-biased and lack outcomes. "Validity" is a
   self-consistent balancing check, not experimental truth. Without prospective data (or at
   least A-Lab outcome data, which the schema can't currently hold), claims about
   *realizability* are unsupported. *Mitigation: add outcome fields; pursue even small
   prospective validation; use A-Lab published outcomes where possible.*

6. **Thermodynamics is mostly heuristic.** Gas-release and class-matching proxies are weakly
   correlated with real driving forces; the MP path is optional and untested at scale; MACE
   (named in the proposal) is absent. The thermo term (`w_t = 0.6`) may be adding noise rather
   than physics. *Mitigation: ablate thermo (switch exists conceptually via scoring weights),
   and add cached MP hull energies or a MACE surrogate before claiming physics grounding.*

**Bottom line on validity:** The *engineering* is valid and the *design* is reasonable, but
the *scientific validity of the central claim is unproven and structurally at risk*. The
project needs to (1) produce real, leakage-controlled results, (2) show search adds value over
retrieval, and (3) put the judge on a non-circular calibration footing before any strong claim
— let alone a Nature-tier one — is warranted.

---

## Prioritized Recommendations

1. **[Blocking] Run the real pipeline and publish numbers.** Real data → all four splits →
   MCTS vs NN vs frequency vs ablations, multiple seeds, with CIs. This is the single most
   important missing thing; without it there is no science, only scaffolding.
2. **[Blocking] Prove or disprove that MCTS beats greedy/NN.** Add beam-search, LLM-only, and
   random baselines; report MCTS−NN on target-disjoint and chemical-system splits. Be
   prepared to enlarge the action space or reframe if the gap is null.
3. **[High] De-circularize judge calibration** using corrupted negatives + held-out recovery,
   and validate against a small expert-rated sample.
4. **[High] Add top-k recall and condition-range metrics; stop reporting `top1_validity` as a
   success rate** (it's ~1.0 by construction).
5. **[High] Enrich the schema and normalization** (amounts, hydrate/role, outcome) and
   **replace substring precursor classification** with composition-based classification.
6. **[Medium] Deepen conditions/actions** with learned per-family distributions so search is
   non-trivial; wire in the already-written partial-state judge.
7. **[Medium] Real physics** (cached MP hulls / MACE surrogate) for driving force and
   competing phases, or drop the thermo term until it is validated.
8. **[Medium] Automated leakage audits** in the benchmark harness (assert train/test DOI and
   target disjointness for the hard splits).
9. **[Lower] Structural-prototype split** (via MP spacegroup/structure) to replace the coarse
   `target_class` "material_family" split; robust publication-year metadata instead of DOI
   regex.
10. **[Lower / high-impact-if-done] Prospective validation** of a few targets with XRD
    outcomes — the difference between a methods paper and a landmark one.
