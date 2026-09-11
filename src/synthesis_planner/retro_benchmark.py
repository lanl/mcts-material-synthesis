"""Leakage-controlled, retrosynthesis-style benchmark (Phase 3 / WS-F).

Mirrors how MCTS retrosynthesis is evaluated (solve-rate + top-k route recovery,
PaRoutes/USPTO-style) adapted to inorganic synthesis:

- **solve_rate**  - fraction of held-out targets for which the top route bottoms
  out entirely in a *train-mined* stock of buildable precursors (the inorganic
  analog of "all leaves purchasable"). This is the headline metric and is NOT
  the self-consistency ``top1_validity`` that EVALUATION.md flagged as ~1.0.
- **precursor_recall@k** - does any of the top-k predicted precursor sets exactly
  match any *gold* literature route for that target (recall over the union of all
  known routes, as PaRoutes does)?
- **class_recall@k**, **precursor_jaccard** - softer precursor-set agreement.
- **condition_in_range** - predicted calcination temperature within the gold
  temperature envelope for the target.

Leakage control: the retrieval index, stock and templates are all mined from the
TRAIN split only, and a split-key audit asserts no held-out target/chemical
system appears in train. All offline (deterministic judge); no LLM/API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Callable

from .benchmark import _first_heating_temperature, _operation_similarity, build_split
from .core.schema import PlannedRoute, PlanningProblem, RouteRecord
from .data.retrieval import RetrievalIndex
from .data.stock import Stock, build_stock, route_is_solved, stock_coverage
from .data.templates import mine_templates
from .planner import SynthesisPlanner

TEMPERATURE_TOLERANCE_C = 75.0


@dataclass
class GoldTarget:
    """The union of all literature routes for one held-out target."""

    target_formula: str
    precursor_sets: list[frozenset[str]]
    class_sets: list[tuple[str, ...]]
    temperatures: list[float]
    routes: list = field(default_factory=list)
    heat_stage_counts: list = field(default_factory=list)


@dataclass
class RetroBenchmarkResult:
    method: str
    split_type: str
    modality: str
    n_train: int
    n_targets: int
    # Headline procedure metrics (the reframing): does the method produce a
    # high-quality, buildable full synthesis PROCEDURE?
    mean_synthesizability: float
    mean_operation_similarity: float
    condition_in_range_rate: float
    stage_count_match_rate: float
    solve_rate: float
    mean_stock_coverage: float
    # Secondary diagnostic: exact precursor-set recovery (retrieval-dominated).
    precursor_recall_at_1: float
    precursor_recall_at_k: float
    class_recall_at_k: float
    mean_precursor_jaccard: float
    top_k: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _precursor_set(route: PlannedRoute | RouteRecord) -> frozenset[str]:
    return frozenset(p.formula for p in route.precursors)


def _class_set(route: PlannedRoute | RouteRecord) -> tuple[str, ...]:
    return tuple(sorted(p.class_name for p in route.precursors))


def build_gold_targets(test_routes: list[RouteRecord]) -> dict[str, GoldTarget]:
    """Group the held-out routes by target into a union of gold recipes."""
    gold: dict[str, GoldTarget] = {}
    for route in test_routes:
        g = gold.get(route.target_formula)
        if g is None:
            g = GoldTarget(route.target_formula, [], [], [])
            gold[route.target_formula] = g
        g.precursor_sets.append(_precursor_set(route))
        g.class_sets.append(_class_set(route))
        g.routes.append(route)
        g.heat_stage_counts.append(_heat_stage_count(route.operations))
        temp = _first_heating_temperature(route.operations)
        if temp is not None:
            g.temperatures.append(temp)
    return gold


def _heat_stage_count(operations) -> int:
    return sum(1 for op in operations if op.verb == "heat")


def _synthesizability(top: PlannedRoute, stock: Stock) -> float:
    """Bounded [0,1] offline synthesizability proxy for a produced procedure.

    Mean of the route's own chemistry/condition quality components plus how much
    of it bottoms out in stock - i.e. "is this a good, buildable procedure?",
    independent of whether it exactly matches a literature recipe.
    """
    s = top.score
    parts = [s.stoich, s.thermo, s.condition, stock_coverage(top.precursors, stock)]
    return sum(parts) / len(parts)


def _best_operation_similarity(top: PlannedRoute, gold: GoldTarget) -> float:
    if not gold.routes:
        return 0.0
    return max(_operation_similarity(top, g) for g in gold.routes)


def _stage_count_matches(top: PlannedRoute, gold: GoldTarget) -> bool:
    return _heat_stage_count(top.operations) in set(gold.heat_stage_counts)


def leakage_audit(
    train_routes: list[RouteRecord], test_routes: list[RouteRecord], split_type: str
) -> dict:
    """Assert the held-out split-keys never appear in train; return a report."""
    key: Callable[[RouteRecord], object]
    if split_type == "chemical_system":
        key = lambda r: r.chemical_system
    elif split_type == "target_formula":
        key = lambda r: r.target_formula
    else:
        key = lambda r: r.route_id
    train_keys = {key(r) for r in train_routes}
    test_keys = {key(r) for r in test_routes}
    overlap = train_keys & test_keys
    report = {
        "split_type": split_type,
        "n_train_keys": len(train_keys),
        "n_test_keys": len(test_keys),
        "leaked_keys": sorted(map(str, overlap))[:20],
        "n_leaked": len(overlap),
        "clean": len(overlap) == 0,
    }
    return report


def _recall_metrics(predictions: list[PlannedRoute], gold: GoldTarget) -> dict:
    pred_sets = [_precursor_set(p) for p in predictions]
    pred_classes = [_class_set(p) for p in predictions]
    gold_sets = gold.precursor_sets
    gold_classes = set(gold.class_sets)

    recall_at_1 = bool(pred_sets and pred_sets[0] in gold_sets)
    recall_at_k = any(ps in gold_sets for ps in pred_sets)
    class_recall_at_k = any(pc in gold_classes for pc in pred_classes)

    # Best formula-level Jaccard of the top prediction against any gold set.
    best_jaccard = 0.0
    if pred_sets:
        top = pred_sets[0]
        for gs in gold_sets:
            union = top | gs
            if union:
                best_jaccard = max(best_jaccard, len(top & gs) / len(union))
    return {
        "recall_at_1": recall_at_1,
        "recall_at_k": recall_at_k,
        "class_recall_at_k": class_recall_at_k,
        "jaccard": best_jaccard,
    }


def _condition_in_range(top: PlannedRoute, gold: GoldTarget) -> bool | None:
    if not gold.temperatures:
        return None
    pred_temp = _first_heating_temperature(top.operations)
    if pred_temp is None:
        return False
    lo = min(gold.temperatures) - TEMPERATURE_TOLERANCE_C
    hi = max(gold.temperatures) + TEMPERATURE_TOLERANCE_C
    return lo <= pred_temp <= hi


def _predict(
    method: str,
    planner: SynthesisPlanner,
    problem: PlanningProblem,
    train_routes: list[RouteRecord],
    stock: Stock,
    templates,
    retrieval: RetrievalIndex,
    iterations: int,
    rollout_count: int,
    top_k: int,
    seed: int,
    rng,
) -> list[PlannedRoute]:
    """Dispatch a prediction method, all conditioned on TRAIN-only artifacts."""
    if method == "mcts":
        return planner.plan_with_routes(
            problem, train_routes, iterations=iterations, top_k=top_k,
            rollout_count=rollout_count, seed=seed, stock=stock, templates=templates,
            retrieval=retrieval,
        )
    if method == "nearest_neighbor":
        return planner.plan_nearest_neighbor(problem, train_routes, top_k=top_k)
    if method == "frequency_prior":
        return planner.plan_frequency_prior(problem, train_routes, top_k=top_k)
    if method == "random":
        # Lower bound: score k random train routes as if proposed for this target.
        sample = rng.sample(train_routes, min(top_k, len(train_routes)))
        return [
            planner.score_route_record(problem, route, train_routes, use_judge=False, judge_name="none")
            for route in sample
        ]
    raise ValueError(f"Unknown method: {method}")


def run_retro_benchmark(
    processed_dir: str,
    modality: str = "solid_state",
    split_type: str = "chemical_system",
    methods: tuple[str, ...] = ("mcts", "nearest_neighbor", "frequency_prior", "random"),
    test_fraction: float = 0.2,
    max_targets: int | None = 80,
    iterations: int = 60,
    rollout_count: int = 3,
    top_k: int = 3,
    min_frequency: int = 20,
    seed: int = 0,
) -> dict:
    """Run the leakage-controlled retro benchmark and return a results dict."""
    import random

    from .benchmark import load_routes

    routes = load_routes(processed_dir, modality)
    train_routes, test_routes = build_split(routes, split_type, test_fraction=test_fraction, seed=seed)
    audit = leakage_audit(train_routes, test_routes, split_type)

    # Train-only artifacts (the core of leakage control).
    stock = build_stock(train_routes, min_frequency=min_frequency)
    templates = mine_templates(train_routes)
    retrieval = RetrievalIndex(train_routes)

    gold = build_gold_targets(test_routes)
    # Restrict to targets whose formula the chemistry engine can actually parse;
    # solid-solution / variable-stoichiometry notation (e.g. "(Mn0.5Co0.5)3O4")
    # is not balanceable and is a known data-quality limitation.
    from .core.formula import parse_formula

    def _parseable(formula: str) -> bool:
        try:
            parse_formula(formula)
            return True
        except Exception:
            return False

    all_targets = sorted(gold)
    targets = [t for t in all_targets if _parseable(t)]
    n_unparseable = len(all_targets) - len(targets)
    rng = random.Random(seed)
    if max_targets is not None and len(targets) > max_targets:
        targets = rng.sample(targets, max_targets)
        targets.sort()

    planner = SynthesisPlanner(processed_dir=processed_dir)
    results: dict[str, dict] = {
        "_leakage_audit": audit,
        "_n_targets": len(targets),
        "_n_unparseable_targets_skipped": n_unparseable,
        "methods": {},
    }

    for method in methods:
        synth, opsim, stage_match = [], [], []
        solved, coverage, r1, rk, ck, jac = [], [], [], [], [], []
        cond_hits, cond_total = 0, 0
        for idx, target in enumerate(targets):
            problem = PlanningProblem(target_formula=target, modality=modality)
            try:
                preds = _predict(
                    method, planner, problem, train_routes, stock, templates, retrieval,
                    iterations, rollout_count, top_k, seed + idx, rng,
                )
            except Exception:
                # A pathological target recipe (un-balanceable sub-formula) counts
                # as an unsolved miss rather than aborting the whole benchmark.
                preds = []
            if not preds:
                synth.append(0.0); opsim.append(0.0); stage_match.append(0.0)
                solved.append(0.0); coverage.append(0.0); r1.append(0.0)
                rk.append(0.0); ck.append(0.0); jac.append(0.0)
                continue
            top = preds[0]
            g = gold[target]
            # Headline: procedure quality.
            synth.append(_synthesizability(top, stock))
            opsim.append(_best_operation_similarity(top, g))
            stage_match.append(1.0 if _stage_count_matches(top, g) else 0.0)
            in_range = _condition_in_range(top, g)
            if in_range is not None:
                cond_total += 1
                cond_hits += 1 if in_range else 0
            # Buildability + secondary recovery diagnostics.
            solved.append(1.0 if route_is_solved(top.precursors, stock) else 0.0)
            coverage.append(stock_coverage(top.precursors, stock))
            m = _recall_metrics(preds, g)
            r1.append(1.0 if m["recall_at_1"] else 0.0)
            rk.append(1.0 if m["recall_at_k"] else 0.0)
            ck.append(1.0 if m["class_recall_at_k"] else 0.0)
            jac.append(m["jaccard"])

        res = RetroBenchmarkResult(
            method=method, split_type=split_type, modality=modality,
            n_train=len(train_routes), n_targets=len(targets),
            mean_synthesizability=mean(synth) if synth else 0.0,
            mean_operation_similarity=mean(opsim) if opsim else 0.0,
            condition_in_range_rate=(cond_hits / cond_total) if cond_total else 0.0,
            stage_count_match_rate=mean(stage_match) if stage_match else 0.0,
            solve_rate=mean(solved) if solved else 0.0,
            mean_stock_coverage=mean(coverage) if coverage else 0.0,
            precursor_recall_at_1=mean(r1) if r1 else 0.0,
            precursor_recall_at_k=mean(rk) if rk else 0.0,
            class_recall_at_k=mean(ck) if ck else 0.0,
            mean_precursor_jaccard=mean(jac) if jac else 0.0,
            top_k=top_k,
        )
        results["methods"][method] = res.to_dict()
    return results
