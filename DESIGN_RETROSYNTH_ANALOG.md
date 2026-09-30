# DESIGN: MCTS for Materials Synthesis as a Rigorous Analog of MCTS Retrosynthesis

**Status:** Design / research deliverable. No source code is modified by this document.
**Scope:** How to turn the current single-step forward planner into a recursive,
stock-grounded, retrosynthesis-style planner and benchmark it the way organic
retrosynthesis is benchmarked.
**Audience:** Implementation agents (see §4 for the workstream split).

---

## 0. TL;DR

The current engine (`core/mcts.py` + `core/grammar.py`) is a **single-target, forward,
4-stage template filler**: `precursors -> preparation -> heating -> finalize`. Every leaf
is a complete recipe for the *one* root target, and `top1_validity` is ~1.0 by construction
because the grammar only emits balanceable recipes and the same balancer checks them
(`EVALUATION.md` §4, §6). There is no "stock", no terminal/solved criterion, no recursion,
and no held-out route-recovery benchmark.

The retrosynthesis analog reframes the problem as: **decompose a target into a synthesis DAG
whose leaves are all in a `stock` of buildable precursors.** The transferable machinery is
(a) a stock/terminal criterion, (b) a backward expansion policy that proposes precursor sets
(= "reactions applied backward"), (c) recursion (a proposed precursor not in stock becomes a
sub-target), (d) a value that rewards *reaching stock* rather than self-consistency, and
(e) route-recovery metrics against held-out literature routes.

**The single most important honest finding from the data** (measured on
`data/processed/solid_state_routes.jsonl`, 19,488 routes): only **0.6 % of precursor
occurrences** use a precursor that is itself a target elsewhere in the corpus. Inorganic
solid-state synthesis is overwhelmingly **single-step from stock oxides/carbonates**. Only
**179 distinct precursor formulas** (BaTiO3 ×70, SrTiO3 ×32, CaTiO3, Li2SiO3, BaZrO3, …) are
ever synthesized as targets. So the retrosynthesis analogy is real but **shallow**: the
realistic near-term win is enabling **depth-1 and occasional depth-2** DAGs plus a
principled solved-criterion, *not* the deep 5-10 step trees of organic CASP. The design
below is engineered around that reality rather than against it.

---

## 1. Component mapping: organic retrosynthesis ↔ materials synthesis ↔ this repo

| Organic retrosynthesis concept | Materials-synthesis analog | Where it lives / should live in THIS repo | Disanalogy & handling |
|---|---|---|---|
| **Node = molecule (target / intermediate)** | Node = target *material* (a formula + modality) that still needs to be made | New: `SubTarget` inside a `SynthesisDAG` node; `PlanningState.problem.target_formula` already carries a formula. Today a node is a *stage within one recipe*, not a material. | Materials have no canonical SMILES; identity = normalized formula (`core/formula.py normalized_composition`) + optional structure/phase. Solid solutions/dopants (`(Y0.9Ca0.1)BaCo2O5.5`) are dropped by `filter_dataset.py` — keep dropping for v1, flag as known gap. |
| **Edge = reaction template applied backward** | "Decomposition template" = target-class → precursor-class pattern + a condition schedule (e.g. `perovskite ABO3 <= A-carbonate + B-oxide, calcine 900-1300 C air`) | New `data/templates.py` mining backward templates from the two JSONL corpora; consumed by a rewritten `grammar.expand_state`. Today `grammar._heating_actions` hard-codes schedules; precursor sets come from `retrieval.candidate_precursor_sets`. | Inorganic "reactions" are fuzzy: no atom-mapping, extraction-noisy stoichiometry, multiple valid routes. Handle by making a template a *distribution over precursor-class multisets + condition ranges conditioned on target chemistry*, not an exact graph-rewrite rule. Validity is enforced downstream by the real balancer (`chemistry.balance_route`), which is the strongest asset. |
| **Expansion policy network** (which templates to apply) | Retrieval + frequency priors over precursor-class multisets, conditioned on target elements/class | `data/retrieval.py candidate_precursor_sets` + `precursor_usage_by_element`/`class_usage_by_element` already do this per-element. Extend to emit *class-pattern* templates and per-family condition priors. | No trained neural policy (offline, no GPU assumed). Use empirical priors + optional lightweight logistic/GBM policy as a stretch. This is acceptable — AiZynthFinder-style search works with 100 iterations and expansion priors. |
| **Rollout / filter policy** | Random/greedy prior-weighted completion of remaining stages + hard-constraint filter | `grammar.rollout_completion`; filter = `constraints.evaluate_hard_constraints` (coverage, balance, redox, atmosphere). | Segler 2018 uses a *filter network* to kill infeasible reactions; here the deterministic balancer + redox analyzer is a stronger, exact filter. Keep it; it is the analog of the filter net. |
| **Value / reward** | Reward for a route **all of whose leaves are in stock** + validity/thermo/condition plausibility | `core/scoring.py evaluate_state` — must be redesigned (§2.4) so the dominant term is *solved-to-stock*, not `retrieval`/`judge` self-consistency. | Today `mcts_value = total` mixes retrieval self-similarity; leads to `top1_validity≈1.0` circularity. Fix: gate reward on solved-criterion + recovery-oriented terms. |
| **Stock / purchasable building blocks (terminal criterion)** | A mined **stock set** of common buildable precursors (oxides/carbonates/nitrates/metals) | New `data/stock.py` mining stock from precursor frequency; queried by a new `is_in_stock(formula)`; a route is *solved* when every leaf precursor ∈ stock. | In organics the stock is a vendor catalog; here we mine it from precursor frequency (§2.1). A stock membership function replaces "purchasable". |
| **Recursion → multi-step route / DAG** | A proposed precursor **not** in stock becomes a new sub-target expanded by the same grammar | New `SynthesisDAG` orchestration wrapping `MonteCarloTreeSearch` (§2.2). Today there is no recursion; `PlanningState.is_terminal` just checks `stage=="terminal"`. | Recursion is rare (0.6 %) and shallow (depth ≤2 in practice). Cap `max_depth=2-3`; expect most routes to stay depth-1. This is the central disanalogy — see §5. |
| **Solve rate; n1/n5 route accuracy; top-k; route length; #nodes/time** (PaRoutes; USPTO-50K) | Same metrics adapted: solve-rate vs stock, precursor-set recall@k, class recall@k, condition-in-range accuracy, DAG route length, nodes/time | New metrics in `benchmark.py`; splits already exist in `benchmark.build_split`. | "Exact route" is ill-defined (multiple valid routes, noisy extraction). Use **set/multiset recall + tree-edit-distance-lite on the DAG** rather than exact string match (§3). |

### The hard disanalogies, stated plainly
1. **Depth.** Organic CASP routinely needs 5-10 backward steps; inorganic solid-state is ~99 % one step (measured). The tree is shallow → MCTS has less to search than in CASP. *Handle:* make the *breadth* (precursor chemistry × condition schedule × modality) the search space, and treat recursion as an occasional depth-2 bonus for the 179 recursive precursors, not the headline.
2. **No atom mapping / no exact templates.** *Handle:* templates are class-level patterns + condition distributions; correctness is enforced by the exact `chemistry.balance_route` engine, not by template validity.
3. **Conditions dominate outcome.** Temperature/atmosphere/dwell decide phase purity far more than in organics. *Handle:* conditions are first-class search dimensions with per-family learned ranges, and condition-in-range is a scored metric.
4. **No true negatives / success labels.** Literature is success-biased; the schema has no `outcome`. *Handle:* generate *corrupted-route negatives* for judge calibration (§3.5); do not claim realizability.
5. **Stock is not a catalog.** *Handle:* mine and version a stock set with an explicit, reported definition (§2.1); report sensitivity to the stock threshold.

---

## 2. Core architectural redesign (file/function-referenced)

### 2.1 Stock definition and terminal / solved criterion

**New module `src/synthesis_planner/data/stock.py`.**

Mine the stock from precursor frequency across both processed corpora. Measured
distribution (solid-state) already supports a clean cut: `TiO2` (3811), `SrCO3` (3239),
`BaCO3` (3170), `La2O3`, `CaCO3`, `Bi2O3`, `Fe2O3`, `Nb2O5`, `Li2CO3`, … with class
distribution `oxide` 41,921 / `carbonate` 14,652 / `elemental_or_other` 2,686 /
`nitrate` 1,755 / `hydroxide` 697 / `halide` 696 / `acetate` 499 / `sulfide` 264.

```python
# data/stock.py  (design sketch)
@dataclass(frozen=True)
class Stock:
    formulas: frozenset[str]          # explicit high-frequency buildable precursors
    allowed_classes: frozenset[str]   # {"oxide","carbonate","nitrate","hydroxide",
                                      #  "acetate","sulfate","halide","elemental_or_other"}
    min_frequency: int                # threshold used to mine `formulas`

    def contains(self, precursor: PrecursorRecord) -> bool:
        # A precursor is "in stock" if it is a frequently-used commodity precursor
        # (formula in `formulas`) OR belongs to a commodity class and is a simple
        # (binary/unary + common anion) compound. Complex multi-cation oxides are NOT
        # in stock -> they can become sub-targets.
        ...

def build_stock(routes, min_frequency=20, top_n_per_element=..., ) -> Stock: ...
def load_or_build_stock(processed_dir, ...) -> Stock: ...   # cache to data/processed/stock.json
```

Concretely, the *buildable set* = precursor formulas with corpus frequency ≥ `min_frequency`
(≈179 formulas at ≥20) **plus** a rule: any binary oxide `MOx`, simple carbonate/nitrate/
hydroxide/acetate of a single cation, or elemental metal is in stock. **Not** in stock:
multi-cation complex oxides (perovskites, spinels, garnets) — these are the recursion
candidates and coincide with the 179 formulas that appear as both precursor and target
(BaTiO3, SrTiO3, YBa2Cu3O7, ZnFe2O4, …).

**Solved criterion.** Add to `PlanningState` (or the new DAG wrapper) a notion:
a leaf precursor is *terminal* iff `stock.contains(precursor)`. A full DAG is **solved** iff
every leaf across all sub-targets is in stock AND every internal node passes hard checks. This
is the direct analog of AiZynthFinder's "terminated because a set of purchasable building
blocks is reached." Store as `SynthesisDAG.is_solved`.

### 2.2 Recursive, multi-step expansion → synthesis DAG

Today the search is flat: `MonteCarloTreeSearch.run(root_state, ...)` builds one tree over
stages of one target and `PlanningState.is_terminal` is just `stage=="terminal"`
(`core/schema.py:213`). To get a DAG we add an **orchestration layer above** the existing
single-target MCTS rather than rewriting the tree node semantics (keeps 89 tests green).

**New module `src/synthesis_planner/core/retro_planner.py`** (or extend `planner.py`):

```python
@dataclass
class DAGNode:
    target_formula: str
    modality: str
    depth: int
    recipe: PlannedRoute | None          # the chosen single-step recipe for this node
    children: list["DAGNode"]            # sub-targets = non-stock precursors expanded further
    in_stock: bool

def plan_retro(problem, routes, stock, max_depth=2, iterations=250, ...) -> SynthesisDAG:
    # 1. Run the EXISTING single-target MCTS for `problem.target_formula`
    #    -> obtain top single-step recipe(s) via planner.plan_with_routes.
    # 2. For each precursor in the chosen recipe:
    #       if stock.contains(p): mark leaf (terminal, in stock)
    #       elif depth < max_depth and p is a known/plausible target:
    #             recurse: plan_retro(sub_problem(p), routes, stock, depth+1)
    #       else: mark as UNSOLVED leaf (dangling precursor)
    # 3. DAG solved iff all leaves in stock.
```

Key design decisions:
- **Reuse, don't rewrite, the inner MCTS.** The inner tree still searches
  precursors×prep×conditions for a *single* material (unchanged `grammar.py` for the forward
  recipe of each node). The DAG recursion is a thin driver. This isolates risk and preserves
  the test suite.
- **Cycle / repeat guard.** Track visited formulas on the path; never expand a formula already
  an ancestor (prevents A←B←A). Analog of AiZynthFinder's loop avoidance.
- **Depth cap.** `max_depth=2` default (data says depth-3 is essentially nonexistent). Beyond
  cap, unsolved precursors are dangling leaves and the DAG is "unsolved" unless they are in
  stock.
- **`PlanningState` change (minimal):** add an optional field `stock_context` (or pass stock
  through `EvaluationConfig`) so the *inner* scorer can reward recipes whose precursors are in
  stock (see 2.4). No change to the frozen-dataclass discipline; add a new frozen field with a
  default so existing constructions in tests keep working. Add `SynthesisDAG`/`DAGNode`
  dataclasses to `core/schema.py`.

### 2.3 Template mining (backward decomposition templates + condition schedules)

**New module `src/synthesis_planner/data/templates.py`.** Mine, per target family, the
empirical distribution of:
- **Precursor-class multiset patterns**: e.g. for `oxide` perovskite ABO3, the mined pattern
  is `{A: carbonate|oxide, B: oxide}` with frequencies. Key by
  `(target_class, sorted cation set signature)` and/or by chemical system.
- **Condition schedules**: temperature percentiles (already partially done in
  `grammar._temperature_setpoints` and `_heating_actions`), dwell, atmosphere, #regrind steps,
  ramp — conditioned on target family, learned from the analog set rather than global medians.
- **Operation-sequence priors**: the ordering (mix→grind→calcine→regrind→sinter) frequencies,
  which `EVALUATION.md` §2 flags as *not currently learned*.

These templates become the **expansion policy**: `grammar.expand_state` at the `precursors`
stage should propose class-pattern-consistent precursor sets (from `templates.py`), ranked by
mined prior × retrieval score, instead of only `retrieval.candidate_precursor_sets`. Condition
stages draw from per-family schedules. This directly deepens the action space so search matters
(the `EVALUATION.md` §3 "MCTS ≈ NN" risk).

Implementation note: this is mostly *aggregation over the existing JSONL* — cheap, offline,
deterministic. Cache to `data/processed/templates.json`.

### 2.4 Reward / value redesign (kill the top1_validity≈1.0 circularity)

Current `scoring.evaluate_state` (`core/scoring.py:14-73`) sums validity(1.5) + stoich(1.2) +
precursor(1.0) + thermo(0.6) + retrieval(1.0) + condition(0.8) + judge(1.0) − penalties, and
`mcts_value = total`. Because the grammar only emits balanceable recipes, `validity`≈1 always,
and `retrieval` rewards similarity to the corpus that generated the candidate → self-consistency.

**Redesign (retrosynthesis-faithful):**
1. **Solved-to-stock is the dominant terminal reward.** Add a term `stock_score` = fraction of
   leaf precursors in stock (1.0 only when the whole (sub)route bottoms out in stock). At the
   DAG level, `solved` (all leaves in stock, all nodes valid) yields the large reward; partial
   solve yields proportional credit. This is the analog of "route found to purchasable stock."
2. **Demote self-consistency terms.** Cut `retrieval` weight sharply (it should *guide*
   expansion via priors, not *reward* the leaf). Keep validity/stoich/redox as **hard gates**
   (invalid ⇒ large negative, as now) but **stop reporting `validity` as a success metric**.
3. **Condition plausibility** stays (condition-in-range vs mined family ranges) — it is a real,
   non-circular signal.
4. **Value backup.** Keep the `value_aggregation="mean"` default in `core/mcts.py` (already the
   better choice per the docstring); the "max" optimism is a diagnostic only.
5. **Report solve-rate, not validity.** The headline number becomes "fraction of held-out
   targets for which a fully-in-stock, valid DAG is found," exactly mirroring PaRoutes solve-rate.

This makes the reward reward *reaching stock*, not agreeing with retrieval — the core fix
`EVALUATION.md` demands.

### 2.5 Selection / rollout policy changes to `core/mcts.py`

Mostly **no structural change needed** — the PUCT engine (`_select`, `_expand`, `_simulate`,
`_backup`, `_puct_score`) is sound and stays. Required touches:
- `_simulate` already averages rollouts (mean) — good. Ensure `evaluate_state` receives the
  stock context so rollout values reflect solved-to-stock (pass `stock` via `EvaluationConfig`).
- Optionally add **progressive widening** if templates make the precursor-set branching large,
  but at depth-1 with ≤12 candidate sets this is unnecessary for v1.
- The DAG recursion lives *outside* `mcts.py` in `retro_planner.py`; `mcts.py` stays a
  single-material solver. This is the lowest-risk way to add recursion.

---

## 3. Benchmark & evaluation spec (mirroring retrosynthesis)

Retrosynthesis practice we mirror: **single-step top-k precursor recall** (USPTO-50K style) and
**multi-step solve-rate + top-k route accuracy via tree-edit-distance** (PaRoutes N1/N5 style).
See references in §6.

### 3.1 Metrics
Add to `benchmark.py` (`BenchmarkSummary`, `evaluate_split`):
- **Solve rate** — fraction of held-out targets for which the planner returns a *solved* DAG
  (all leaves ∈ stock, all nodes valid). *This is the new headline metric.* NEW.
- **Precursor-set recall@k** — is the gold precursor set (as a set of formulas) among the top-k
  predicted single-step precursor sets? (Relaxes the current exact@1 in
  `_precursor_exact_match`, `benchmark.py:288`.) NEW.
- **Precursor-class recall@k** — same on `class_name` multisets (relax `benchmark.py:292`). NEW.
- **Condition-in-range accuracy** — predicted first-heating temperature within the mined
  family range (not just abs error `_temperature_error`, `benchmark.py:318`). NEW.
- **Route length / DAG depth & node count** — distribution of DAG depth and #nodes. NEW.
- **Search efficiency** — iterations, wall-clock, #`evaluate_state` calls per solve. NEW.
- **Route recovery (top-k route accuracy)** — see 3.2. NEW.
- Keep existing: `operation_similarity` (`SequenceMatcher`, good), `temperature_error_c`,
  `analog_support`. **Retire `top1_validity_rate` as a success metric** (report it only as a
  sanity check; it is ~1.0 by construction — `EVALUATION.md` §4).

### 3.2 How "route recovery" is computed for inorganic routes (given the schema)
Organic PaRoutes uses Tree Edit Distance on the route tree. Our inorganic DAGs are shallow, so
define a **graded route-match score** between predicted DAG and the gold `RouteRecord`:
- **Depth-1 (the common case):** route match = precursor-set match. Report exact set match,
  Jaccard on formulas, and class-multiset match. This *is* the single-step recall metric.
- **Depth ≥2:** compare DAGs by a lightweight tree-edit distance over `(formula, precursor-set)`
  nodes: cost 1 per node insert/delete, cost = (1 − precursor-set Jaccard) per node substitution;
  normalize by max node count. TED = 0 ⇒ exact route recovery. Since the gold corpus is almost
  all depth-1, most gold routes are single nodes; a predicted depth-2 route "recovers" the gold
  if flattening its leaf precursors reproduces (superset/subset within tolerance) the gold
  precursor set. Report both strict and flattened recovery.
- Because multiple routes are valid (proposal §10), report **recall@k over the union of *all*
  gold routes for that target/chemical system**, not a single gold — mirroring PaRoutes N5's
  multiple-routes-per-target treatment.

### 3.3 Splits & leakage control
`benchmark.build_split` already supports `target_formula`, `chemical_system`,
`material_family` (coarse), `publication_year`, `random` (`benchmark.py:84-119`). Use:
- **Headline splits:** `chemical_system` (hardest, tests extrapolation) and `target_formula`.
  Report `random` only as an upper bound.
- **Leakage control (must add):**
  - The retrieval index is already built from `train_routes` only (`benchmark.py:140`) — good.
    Add an **explicit leakage audit**: assert no test `route_id`/`source_doi`/`target_formula`
    (and for chemical_system split, no test chemical_system) appears in the train retrieval
    corpus, template mining set, OR **stock mining set**. NEW — `EVALUATION.md` §6.2 flags this
    is unasserted.
  - **Stock must be mined from train only** per split (a target-disjoint stock), else a held-out
    complex oxide could leak in as "stock". Report stock size per split.
  - **Templates mined from train only.**
  - Emit a `leakage_report` in `BenchmarkSummary`.

### 3.4 Baselines
Present: `nearest_neighbor`, `frequency_prior`, and ablations `mcts_no_retrieval`,
`mcts_no_judge`, `mcts_no_hard_checks` (`benchmark.py:392-445`, `planner.py`). **Add:**
- **`random` baseline** — random in-stock precursor set covering target elements + default
  schedule (null model for solve-rate). NEW.
- **`beam_search`** — beam over the same template/condition priors, no tree search; isolates
  "search vs greedy enumeration" (`EVALUATION.md` §3.2, §4 name this as the key missing figure).
  NEW.
- **`greedy`** — depth-first argmax-prior completion (already ~`plan_frequency_prior`; formalize
  as a named baseline).
- (LLM-only planner is **out of scope** — no API key; note it as future work.)
**Headline figure:** `MCTS_retro − nearest_neighbor` solve-rate and recall@k on `chemical_system`
and `target_formula` splits, multiple seeds, bootstrap CIs. If MCTS does not beat NN + beam, the
thesis must be reframed (honest risk, §5).

### 3.5 Judge calibration (de-circularized, offline)
`judge_calibration.py` currently correlates the deterministic judge against `hard_checks.valid`
on near-all-valid literature routes — circular (`EVALUATION.md` §4, §6.4). Replace with:
- **Corrupted-route negatives:** perturb gold routes (swap a precursor to a wrong element,
  wrong atmosphere, delete the heating step) and check the judge/score ranks gold > corrupted.
  Report AUROC/precision-recall of the score as a filter. Fully offline.
- **Held-out recovery correlation:** does a higher route score predict recovering the true
  precursor set on target-disjoint splits? Report correlation. No LLM needed — the deterministic
  judge + chemistry score is the ranker.

---

## 4. Phased implementation plan (independent workstreams)

Test baseline to preserve: **89 passed / 1 skipped** (`test_materials_project` skips without MP
key). Every workstream must keep the suite green and add its own tests.

```
Ordering / dependency graph:

  WS-A (stock) ──┐
                 ├──► WS-D (reward redesign) ──► WS-E (DAG recursion) ──► WS-F (retro benchmark)
  WS-B (templates)┘                                                        ▲
  WS-C (metrics/leakage/baselines, split-side) ──────────────────────────┘

  WS-A, WS-B, WS-C are INDEPENDENT and parallelizable from day 1.
  WS-D depends on WS-A (needs stock in scoring).
  WS-E depends on WS-A + WS-D (recursion needs solved-criterion + stock-aware reward).
  WS-F depends on WS-C (metrics) + WS-E (DAG to evaluate); can start the single-step
     recall/leakage parts as soon as WS-C lands.
```

### WS-A — Stock mining & solved criterion  *(parallel, foundational)*
- **Scope:** `data/stock.py` (`Stock`, `build_stock`, `load_or_build_stock`, `contains`);
  cache `data/processed/stock.json`; CLI `build-stock` in `cli/main.py`; add `SynthesisDAG`/
  `DAGNode` skeleton dataclasses to `core/schema.py`.
- **Files:** NEW `data/stock.py`; edit `core/schema.py`, `cli/main.py`.
- **Deps:** none. **Test impact:** new `tests/test_stock.py`; no existing test touched.
- **Risk:** low. Main judgment call = the stock threshold; make it a parameter and report
  sensitivity.

### WS-B — Template & condition-schedule mining  *(parallel)*
- **Scope:** `data/templates.py` mining class-pattern precursor templates + per-family condition
  distributions + operation-sequence priors; cache `data/processed/templates.json`.
- **Files:** NEW `data/templates.py`; later consumed by `grammar.py` (WS-E).
- **Deps:** none (reads processed JSONL). **Test impact:** new `tests/test_templates.py`.
- **Risk:** low-moderate. Precursor-class labels are noisy substring heuristics
  (`datasets.classify_precursor`, `EVALUATION.md` §2) — templates inherit that noise; consider a
  composition-based reclassification as a stretch, but v1 can use existing classes.

### WS-C — Metrics, leakage audit, split-side baselines  *(parallel)*
- **Scope:** add recall@k, class-recall@k, condition-in-range, DAG-length/#nodes/time fields to
  `BenchmarkSummary`; leakage audit + `leakage_report`; add `random`, `beam_search`, `greedy`
  baseline methods to `_predict_with_method`; stop treating `top1_validity` as success.
- **Files:** `benchmark.py`, `planner.py` (new baseline methods).
- **Deps:** none for the single-step metrics/baselines. **Test impact:** extend
  `tests/test_benchmark.py`, `tests/test_planner.py`.
- **Risk:** low. Keep old fields (default values) so existing benchmark tests still pass.

### WS-D — Reward / value redesign  *(after WS-A)*
- **Scope:** add `stock_score`/solved-to-stock term to `scoring.evaluate_state`; thread stock
  through `EvaluationConfig`; demote `retrieval` weight; keep hard gates. Add
  de-circularized judge calibration (corrupted negatives) in `judge_calibration.py`.
- **Files:** `core/scoring.py`, `core/schema.py` (`EvaluationConfig` gains a stock/weights
  field), `judge_calibration.py`.
- **Deps:** WS-A. **Test impact:** `tests/test_scoring.py`, `tests/test_judge_calibration.py`
  will need updated expectations — coordinate; this is the most test-sensitive workstream.
- **Risk:** moderate. Changing weights shifts many scored outputs; update fixtures deliberately,
  document the new weighting in CLAUDE.md.

### WS-E — DAG recursion / retro planner  *(after WS-A + WS-D)*
- **Scope:** `core/retro_planner.py` (`plan_retro`, `SynthesisDAG`); grammar consumes templates
  (WS-B) at the precursors + condition stages; cycle guard, depth cap; CLI `plan --retro`.
- **Files:** NEW `core/retro_planner.py`; edit `core/grammar.py` (use templates), `planner.py`,
  `cli/main.py`.
- **Deps:** WS-A, WS-D (and WS-B for templates). **Test impact:** new `tests/test_retro_planner.py`
  (fixture: a two-cation oxide that decomposes to a stock carbonate + a recursive titanate).
- **Risk:** moderate. Recursion is rare in data, so guard against over-expanding; keep
  `max_depth=2`.

### WS-F — Retro benchmark & route recovery  *(after WS-C + WS-E)*
- **Scope:** solve-rate over held-out targets, top-k route recovery (TED-lite for depth≥2,
  set-recall for depth-1), full comparison table MCTS_retro vs NN/beam/greedy/random across
  `chemical_system`/`target_formula`, seeds + bootstrap CIs; save to `benchmark_results/`.
- **Files:** `benchmark.py` (route-recovery + DAG evaluation), `cli/main.py` (`benchmark --retro`).
- **Deps:** WS-C, WS-E. **Test impact:** extend `tests/test_benchmark.py`.
- **Risk:** moderate-high — this is where the science is decided (does search beat NN?).

**Parallelism summary:** WS-A, WS-B, WS-C can be handed to three agents simultaneously. WS-D
starts when WS-A merges. WS-E starts when WS-A+WS-D merge (WS-B feeds it but E can stub templates
until B lands). WS-F is last. Critical path: A → D → E → F.

---

## 5. Risks & honest limits

1. **The analogy is shallow by nature.** 0.6 % recursion means "multi-step materials
   retrosynthesis" is mostly a *reframing* (stock + solved-criterion + honest reward), plus a
   real-but-minor depth-2 capability for ~179 recursive precursors. Do **not** market deep
   synthesis trees. The genuine deliverable is: a stock-grounded solve-rate benchmark, non-
   circular reward, richer breadth search, and correct route-recovery metrics.
2. **MCTS may still ≈ nearest-neighbor.** Depth-1 with ≤12 candidate precursor sets is a tiny
   tree. The added breadth (templates × conditions × modality) is what must make search pay;
   the beam-search baseline (WS-C) will expose whether it does. Be prepared to reframe the
   contribution as "a rigorous stock-grounded benchmark + honest reward" rather than "MCTS wins"
   if the gap is null (this is the real scientific test, per `EVALUATION.md`).
3. **Stock definition is a modeling choice, not ground truth.** Solve-rate is only as meaningful
   as the stock; report sensitivity to the frequency threshold and the exact stock set.
4. **No success labels / no true negatives.** Everything is validated against reported (success-
   biased) recipes; "solved" means "reaches buildable precursors and passes chemistry gates," not
   "will work in a furnace." Judge calibration uses synthetic corrupted negatives, not real
   failures (schema has no `outcome` field — `EVALUATION.md` §5).
5. **Noisy inputs.** Precursor-class labels are substring heuristics; solid-solution/doped
   targets are dropped by `filter_dataset.py`. Templates and class-recall inherit this noise.
6. **Unclear in the repo (not fabricated):** the `develop`/`modularize` branches were not
   inspected (work is on `main`); the solution corpus's per-record `type` field mapping to
   hydrothermal/precipitation was taken from `EVALUATION.md`/CLAUDE.md, not re-verified line-by-
   line here; `is_terminal` semantics are confirmed only for the forward grammar.

**Realistic near-term win:** a leakage-controlled, stock-grounded **solve-rate + precursor
recall@k benchmark** on held-out solid-state targets, with a defensible non-circular reward and a
depth-≤2 DAG capability — i.e., the honest "MCTS-for-materials-synthesis" analog, suitable for a
methods/benchmark venue (npj Comput. Mater. / Digital Discovery), not a Nature-tier claim.

---

## 6. Key references (retrosynthesis machinery we are porting)

- **Segler, Preuss & Waller, *Nature* 555, 604 (2018)** — "Planning chemical syntheses with deep
  neural networks and symbolic AI." 3N-MCTS: expansion policy net, filter net, rollout policy
  net; **stock = purchasable building blocks as the terminal criterion**; 100k iterations/molecule.
- **Genheden et al., *J. Cheminform.* 12:70 (2020)** — **AiZynthFinder**: open-source MCTS
  retrosynthesis; a `stock` package terminates search when purchasable building blocks are
  reached; ~100 iterations; expansion policy + no separate filter/rollout net.
  ([Springer](https://link.springer.com/article/10.1186/s13321-020-00472-1))
- **AiZynthFinder 4.0, *J. Cheminform.* (2024)** — 3 years of industrial learnings.
  ([Springer](https://link.springer.com/article/10.1186/s13321-024-00860-x))
- **Chen et al., ICML 2020 — Retro\*** — neural-guided A* search over the AND/OR retrosynthesis
  graph; solve-rate and route quality on held-out targets.
  ([arXiv](https://arxiv.org/pdf/2006.15820))
- **Genheden & Bjerrum — PaRoutes** — multi-step route benchmark: **N1 (10k routes, one per
  patent) and N5 (≤5 routes/patent, longer, convergent)**; **solve-rate** (target solved iff a
  route with all precursors in stock is found) and **top-k route accuracy via Tree Edit Distance**
  (TED=0 ⇒ exact match). This is the template for §3.1–3.2.
- **USPTO-50K single-step benchmark** — top-k exact-match precursor recall (k=1,3,5,10) is the
  standard single-step metric; the analog here is precursor-set/class recall@k (§3.1).
  ([Enhancing MCTS for Retrosynthesis, JCIM 2025](https://pubs.acs.org/doi/10.1021/acs.jcim.5c00417))

---

## 7. Repo files this design touches (index for implementation agents)

| File | Role now | Change |
|---|---|---|
| `data/stock.py` | — | NEW (WS-A): stock mining + `contains` |
| `data/templates.py` | — | NEW (WS-B): backward templates + condition schedules |
| `core/retro_planner.py` | — | NEW (WS-E): DAG recursion driver |
| `core/schema.py` | dataclasses; `PlanningState.is_terminal` (l.213) | add `SynthesisDAG`,`DAGNode`; stock/weights in `EvaluationConfig` |
| `core/mcts.py` | PUCT single-target engine | mostly unchanged; thread stock via config |
| `core/grammar.py` | forward 4-stage templates (`expand_state` l.11, `_heating_actions` l.136) | consume mined templates + per-family conditions |
| `core/scoring.py` | weighted-sum reward (`evaluate_state` l.14) | add solved-to-stock term; demote retrieval |
| `data/retrieval.py` | analog retrieval + precursor priors (`candidate_precursor_sets` l.52) | feed template-consistent candidate sets |
| `benchmark.py` | splits + metrics (`build_split` l.84, `evaluate_split` l.122, `_predict_with_method` l.392) | recall@k, condition-in-range, solve-rate, route recovery, leakage audit, beam/greedy/random baselines |
| `planner.py` | portfolio + baselines (`plan_with_routes` l.63) | add `plan_retro` hook, beam/greedy/random |
| `judge_calibration.py` | circular calibration | corrupted-negative + held-out-recovery calibration |
| `cli/main.py` | argparse subcommands | `build-stock`, `plan --retro`, `benchmark --retro` |
```
