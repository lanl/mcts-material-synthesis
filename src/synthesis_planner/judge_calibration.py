"""Judge calibration and evaluation metrics.

De-circularized per DESIGN_RETROSYNTH_ANALOG.md §3.5 / EVALUATION.md §6.4. The
old routine correlated the judge score against ``hard_checks.valid`` on
literature-only (near-all-valid) routes, which is circular (the deterministic
judge derives from validity) and negatives-free. It is replaced by two fully
offline, non-circular signals:

1. **Corrupted-route negatives.** Each gold route is perturbed three ways -- swap
   a precursor to a wrong (non-covering) element, swap the heating atmosphere,
   and delete the heating step -- and we check that the ranker scores the gold
   route above its corruptions. We report ranking accuracy, mean gold-minus-
   corrupted separation, and the AUROC of the ranker as a gold/corrupted filter.
2. **Held-out recovery correlation.** Over the pool of {gold + corrupted}
   candidates we correlate the ranker score with precursor recovery (match to the
   gold precursor set). A positive correlation means a higher score predicts
   recovering the true precursors -- the property calibration should establish.

The ranker is the deterministic judge + exact chemistry score
(``evaluate_state(...).score.total``), so validity/redox/balance gates and the
demoted retrieval term all feed the ranking. No LLM / API key required.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from statistics import mean

from .core.constraints import evaluate_hard_constraints
from .core.judge import build_judge
from .core.scoring import evaluate_state
from .data.retrieval import RetrievalIndex
from .core.schema import (
    EvaluationConfig,
    NumericRange,
    OperationRecord,
    PlanningProblem,
    PlanningState,
    PrecursorRecord,
    RouteRecord,
)


@dataclass(frozen=True)
class CalibrationResult:
    """Results of de-circularized judge calibration."""

    judge_name: str
    n_samples: int
    n_negatives: int
    corrupted_ranking_accuracy: float   # fraction of corruptions ranked below gold
    ranker_auroc: float                 # AUROC separating gold (pos) from corrupted (neg)
    gold_minus_corrupted: float         # mean(gold score) - mean(corrupted score)
    recovery_correlation: float         # Spearman(score, precursor recovery) over the pool
    mean_gold_score: float
    mean_corrupted_score: float
    mean_judge_score: float             # deterministic judge score on gold (0-1)
    score_distribution: dict[str, int] = field(default_factory=dict)
    correlation_with_validity: float = 0.0  # deprecated; kept for back-compat

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationSample:
    """Single (gold, corrupted-negatives) calibration record."""

    target_formula: str
    gold_score: float
    corrupted_scores: tuple[float, ...]
    gold_judge_score: float
    gold_precursor_match: float


def _score_route(state: PlanningState, analogs, judge_name: str, judge_config: dict | None) -> float:
    """Ranker: deterministic judge + exact chemistry score for a candidate route."""
    planned = evaluate_state(
        state,
        analogs,
        EvaluationConfig(judge_name=judge_name, judge_config=judge_config or {}),
    )
    return planned.score.total


def _corrupt_route(route: RouteRecord) -> list[tuple[str, PlanningState]]:
    """Generate corrupted-negative planning states from a gold route.

    Three perturbations mirroring the design: (1) swap a precursor to a wrong,
    non-covering element; (2) swap the heating atmosphere; (3) delete the heating
    step. Corruptions that cannot be formed (e.g. no heating step to delete) are
    skipped.
    """
    negatives: list[tuple[str, PlanningState]] = []
    base = _route_to_state(route)

    # (1) swap a precursor for an unrelated, non-covering commodity.
    if base.precursors:
        wrong = PrecursorRecord(formula="NaCl", class_name="halide", elements=("Na", "Cl"))
        swapped = (wrong,) + tuple(base.precursors[1:])
        negatives.append(("swap_precursor", _with_fields(base, precursors=swapped)))

    # (2) swap the heating atmosphere to an implausible one.
    if any(op.verb == "heat" for op in base.operations):
        new_ops = []
        swapped_any = False
        for op in base.operations:
            if op.verb == "heat" and not swapped_any:
                bad = "vacuum" if (op.atmosphere or "air") != "vacuum" else "H2"
                new_ops.append(
                    OperationRecord(
                        verb=op.verb,
                        temperature_c=op.temperature_c,
                        time_h=op.time_h,
                        atmosphere=bad,
                        source_label=op.source_label,
                    )
                )
                swapped_any = True
            else:
                new_ops.append(op)
        negatives.append(("swap_atmosphere", _with_fields(base, operations=tuple(new_ops))))

        # (3) delete the heating step(s) entirely.
        no_heat = tuple(op for op in base.operations if op.verb != "heat")
        negatives.append(("delete_heating", _with_fields(base, operations=no_heat)))

    return negatives


def _with_fields(state: PlanningState, **changes) -> PlanningState:
    fields = dict(
        problem=state.problem,
        target_elements=state.target_elements,
        target_class=state.target_class,
        stage=state.stage,
        precursors=state.precursors,
        solvents=state.solvents,
        operations=state.operations,
        evidence_dois=state.evidence_dois,
        analog_targets=state.analog_targets,
    )
    fields.update(changes)
    return PlanningState(**fields)


def _auroc(positives: list[float], negatives: list[float]) -> float:
    """AUROC = P(random positive scored above random negative), ties count 0.5."""
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    for p in positives:
        for n in negatives:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def calibrate_judge(
    judge_name: str,
    test_routes: list[RouteRecord],
    train_routes: list[RouteRecord],
    judge_config: dict | None = None,
    max_samples: int = 100,
) -> CalibrationResult:
    """Calibrate the ranker against corrupted-route negatives + recovery.

    Args:
        judge_name: judge to use inside the ranker.
        test_routes: held-out gold routes (perturbed into negatives).
        train_routes: training routes for the retrieval context (leakage-safe).
        judge_config: optional judge configuration.
        max_samples: maximum gold routes to evaluate.
    """
    retrieval = RetrievalIndex(train_routes)
    judge = build_judge(judge_name, judge_config or {})

    samples: list[CalibrationSample] = []
    for route in test_routes[:max_samples]:
        gold_state = _route_to_state(route)
        analogs = retrieval.retrieve(route.target_formula, top_k=12)

        gold_score = _score_route(gold_state, analogs, judge_name, judge_config)
        corrupted = _corrupt_route(route)
        corrupted_scores = tuple(
            _score_route(state, analogs, judge_name, judge_config) for _, state in corrupted
        )

        hard_checks = evaluate_hard_constraints(gold_state)
        gold_judge_score = judge.evaluate(gold_state, analogs, hard_checks).score

        samples.append(
            CalibrationSample(
                target_formula=route.target_formula,
                gold_score=gold_score,
                corrupted_scores=corrupted_scores,
                gold_judge_score=gold_judge_score,
                gold_precursor_match=1.0,  # gold recovers itself by definition
            )
        )

    if not samples:
        return CalibrationResult(
            judge_name=judge_name,
            n_samples=0,
            n_negatives=0,
            corrupted_ranking_accuracy=0.0,
            ranker_auroc=0.0,
            gold_minus_corrupted=0.0,
            recovery_correlation=0.0,
            mean_gold_score=0.0,
            mean_corrupted_score=0.0,
            mean_judge_score=0.0,
            score_distribution={},
            correlation_with_validity=0.0,
        )

    gold_scores = [s.gold_score for s in samples]
    all_corrupted = [c for s in samples for c in s.corrupted_scores]

    # Ranking accuracy: fraction of (gold, corrupted) pairs where gold > corrupted.
    comparisons = 0
    gold_wins = 0
    for s in samples:
        for c in s.corrupted_scores:
            comparisons += 1
            if s.gold_score > c:
                gold_wins += 1
    ranking_accuracy = gold_wins / comparisons if comparisons else 0.0

    # Recovery correlation over the {gold + corrupted} pool: score vs precursor
    # recovery (gold=1.0; corrupted precursor swaps=lower, others keep precursors).
    pool_scores: list[float] = []
    pool_recovery: list[float] = []
    for route, s in zip(test_routes[:max_samples], samples):
        pool_scores.append(s.gold_score)
        pool_recovery.append(1.0)
        for (label, state), score in zip(_corrupt_route(route), s.corrupted_scores):
            pool_scores.append(score)
            pool_recovery.append(_compute_precursor_match(state, route))
    recovery_corr = _spearman_correlation(pool_scores, pool_recovery)

    distribution = Counter()
    for score in gold_scores:
        distribution[_score_bin(score)] += 1

    return CalibrationResult(
        judge_name=judge_name,
        n_samples=len(samples),
        n_negatives=len(all_corrupted),
        corrupted_ranking_accuracy=ranking_accuracy,
        ranker_auroc=_auroc(gold_scores, all_corrupted),
        gold_minus_corrupted=(mean(gold_scores) - mean(all_corrupted)) if all_corrupted else 0.0,
        recovery_correlation=recovery_corr,
        mean_gold_score=mean(gold_scores),
        mean_corrupted_score=mean(all_corrupted) if all_corrupted else 0.0,
        mean_judge_score=mean(s.gold_judge_score for s in samples),
        score_distribution=dict(distribution),
        correlation_with_validity=0.0,
    )


def _score_bin(score: float) -> str:
    edges = [(-2.0, "<0"), (0.0, "0-1"), (1.0, "1-2"), (2.0, "2-3"), (3.0, "3-4")]
    label = ">=4"
    for threshold, name in edges:
        if score < threshold:
            return name
    return label


def _route_to_state(route: RouteRecord) -> PlanningState:
    """Convert RouteRecord to PlanningState for evaluation"""
    return PlanningState(
        problem=PlanningProblem(
            target_formula=route.target_formula,
            modality=route.modality,
        ),
        target_elements=route.target_elements,
        target_class=route.target_class,
        stage="terminal",
        precursors=route.precursors,
        solvents=route.solvents,
        operations=route.operations,
        evidence_dois=(route.source_doi,) if route.source_doi else (),
        analog_targets=(),
    )


def _compute_precursor_match(state: PlanningState, gold: RouteRecord) -> float:
    """
    Compute precursor class match score.
    Returns fraction of gold precursor classes present in state.
    """
    state_classes = set(p.class_name for p in state.precursors if p.class_name)
    gold_classes = set(p.class_name for p in gold.precursors if p.class_name)

    if not gold_classes:
        return 1.0 if not state_classes else 0.0

    return len(state_classes & gold_classes) / len(gold_classes)


def _spearman_correlation(x: list[float], y: list[float]) -> float:
    """
    Compute Spearman rank correlation coefficient.
    Simple implementation without scipy dependency.
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    # Rank x and y
    x_ranks = _rank_data(x)
    y_ranks = _rank_data(y)

    # Compute Pearson correlation on ranks
    n = len(x)
    mean_x = mean(x_ranks)
    mean_y = mean(y_ranks)

    numerator = sum((x_ranks[i] - mean_x) * (y_ranks[i] - mean_y) for i in range(n))
    denominator_x = sum((x_ranks[i] - mean_x) ** 2 for i in range(n)) ** 0.5
    denominator_y = sum((y_ranks[i] - mean_y) ** 2 for i in range(n)) ** 0.5

    if denominator_x == 0 or denominator_y == 0:
        return 0.0

    return numerator / (denominator_x * denominator_y)


def _rank_data(data: list[float]) -> list[float]:
    """Convert data to ranks (1-indexed)"""
    # Create (value, original_index) pairs
    indexed = [(value, i) for i, value in enumerate(data)]
    # Sort by value
    sorted_indexed = sorted(indexed, key=lambda x: x[0])
    # Assign ranks
    ranks = [0.0] * len(data)
    for rank, (value, original_idx) in enumerate(sorted_indexed, start=1):
        ranks[original_idx] = float(rank)
    return ranks


def print_calibration_report(result: CalibrationResult) -> None:
    """Print human-readable calibration report (corrupted-negative based)."""
    print(f"\n{'='*60}")
    print(f"Judge Calibration Report (de-circularized): {result.judge_name}")
    print(f"{'='*60}\n")

    print(f"Gold routes evaluated:   {result.n_samples}")
    print(f"Corrupted negatives:     {result.n_negatives}\n")

    print("Gold vs corrupted-route negatives:")
    print(f"  - Ranking accuracy (gold > corrupted): {result.corrupted_ranking_accuracy:.1%}")
    print(f"  - Ranker AUROC (gold vs corrupted):    {result.ranker_auroc:.3f}")
    print(f"  - Mean gold score:                     {result.mean_gold_score:+.3f}")
    print(f"  - Mean corrupted score:                {result.mean_corrupted_score:+.3f}")
    print(f"  - Separation (gold - corrupted):       {result.gold_minus_corrupted:+.3f}\n")

    print("Held-out recovery:")
    print(f"  - Spearman(score, precursor recovery): {result.recovery_correlation:+.3f}")
    print(f"  - Mean deterministic judge score:      {result.mean_judge_score:.3f}\n")

    if result.n_samples:
        print("Gold score distribution:")
        for bin_range, count in sorted(result.score_distribution.items()):
            bar = '█' * int(count / result.n_samples * 40)
            pct = count / result.n_samples * 100
            print(f"  {bin_range}: {bar} {count:3d} ({pct:4.1f}%)")

    print(f"\n{'='*60}\n")
