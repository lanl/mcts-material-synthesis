"""Judge calibration and evaluation metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Callable

from .constraints import evaluate_hard_constraints
from .judge import build_judge
from .retrieval import RetrievalIndex
from .schema import PlanningProblem, PlanningState, RouteRecord


@dataclass(frozen=True)
class CalibrationResult:
    """Results of judge calibration against ground-truth routes"""
    judge_name: str
    n_samples: int
    correlation_with_validity: float
    correlation_with_precursor_match: float
    correlation_with_element_coverage: float
    mean_judge_score: float
    std_judge_score: float
    high_score_precision: float  # What % of high-scored (>0.7) routes are valid?
    low_score_recall: float      # What % of invalid routes scored low (<0.3)?
    score_distribution: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationSample:
    """Single sample in calibration dataset"""
    target_formula: str
    judge_score: float
    is_valid: bool
    precursor_match: float
    element_coverage: float


def calibrate_judge(
    judge_name: str,
    test_routes: list[RouteRecord],
    train_routes: list[RouteRecord],
    judge_config: dict | None = None,
    max_samples: int = 100,
) -> CalibrationResult:
    """
    Calibrate judge by evaluating held-out ground-truth routes.

    Args:
        judge_name: Name of judge to calibrate
        test_routes: Held-out test routes to evaluate
        train_routes: Training routes for retrieval context
        judge_config: Optional judge configuration
        max_samples: Maximum number of test routes to evaluate

    Returns:
        CalibrationResult with correlation metrics
    """
    judge = build_judge(judge_name, judge_config or {})
    retrieval = RetrievalIndex(train_routes)

    samples = []
    for route in test_routes[:max_samples]:
        # Convert RouteRecord to PlanningState
        state = _route_to_state(route)

        # Get retrieval context
        analogs = retrieval.retrieve(route.target_formula, top_k=12)

        # Evaluate with judge
        hard_checks = evaluate_hard_constraints(state)
        judge_result = judge.evaluate(state, analogs, hard_checks)

        # Compute ground-truth metrics
        is_valid = hard_checks.valid
        precursor_match = _compute_precursor_match(state, route)
        element_coverage = hard_checks.coverage_fraction

        samples.append(CalibrationSample(
            target_formula=route.target_formula,
            judge_score=judge_result.score,
            is_valid=is_valid,
            precursor_match=precursor_match,
            element_coverage=element_coverage,
        ))

    if not samples:
        return CalibrationResult(
            judge_name=judge_name,
            n_samples=0,
            correlation_with_validity=0.0,
            correlation_with_precursor_match=0.0,
            correlation_with_element_coverage=0.0,
            mean_judge_score=0.0,
            std_judge_score=0.0,
            high_score_precision=0.0,
            low_score_recall=0.0,
            score_distribution={},
        )

    # Compute correlations
    validity_corr = _spearman_correlation(
        [s.judge_score for s in samples],
        [1.0 if s.is_valid else 0.0 for s in samples]
    )

    precursor_corr = _spearman_correlation(
        [s.judge_score for s in samples],
        [s.precursor_match for s in samples]
    )

    coverage_corr = _spearman_correlation(
        [s.judge_score for s in samples],
        [s.element_coverage for s in samples]
    )

    # Compute precision and recall
    high_score_samples = [s for s in samples if s.judge_score > 0.7]
    high_score_precision = (
        sum(1 for s in high_score_samples if s.is_valid) / len(high_score_samples)
        if high_score_samples else 0.0
    )

    invalid_samples = [s for s in samples if not s.is_valid]
    low_score_recall = (
        sum(1 for s in invalid_samples if s.judge_score < 0.3) / len(invalid_samples)
        if invalid_samples else 0.0
    )

    # Score distribution
    score_bins = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    distribution = Counter()
    for sample in samples:
        if sample.judge_score < 0.2:
            distribution["0.0-0.2"] += 1
        elif sample.judge_score < 0.4:
            distribution["0.2-0.4"] += 1
        elif sample.judge_score < 0.6:
            distribution["0.4-0.6"] += 1
        elif sample.judge_score < 0.8:
            distribution["0.6-0.8"] += 1
        else:
            distribution["0.8-1.0"] += 1

    # Compute mean and std
    scores = [s.judge_score for s in samples]
    mean_score = mean(scores)
    std_score = (sum((s - mean_score) ** 2 for s in scores) / len(scores)) ** 0.5

    return CalibrationResult(
        judge_name=judge_name,
        n_samples=len(samples),
        correlation_with_validity=validity_corr,
        correlation_with_precursor_match=precursor_corr,
        correlation_with_element_coverage=coverage_corr,
        mean_judge_score=mean_score,
        std_judge_score=std_score,
        high_score_precision=high_score_precision,
        low_score_recall=low_score_recall,
        score_distribution=dict(distribution),
    )


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
    """Print human-readable calibration report"""
    print(f"\n{'='*60}")
    print(f"Judge Calibration Report: {result.judge_name}")
    print(f"{'='*60}\n")

    print(f"Samples evaluated: {result.n_samples}")
    print(f"Mean judge score: {result.mean_judge_score:.3f} ± {result.std_judge_score:.3f}\n")

    print("Correlations with ground truth:")
    print(f"  - Validity:          {result.correlation_with_validity:+.3f}")
    print(f"  - Precursor match:   {result.correlation_with_precursor_match:+.3f}")
    print(f"  - Element coverage:  {result.correlation_with_element_coverage:+.3f}\n")

    print("Precision and Recall:")
    print(f"  - High-score precision (>0.7 → valid): {result.high_score_precision:.1%}")
    print(f"  - Low-score recall (invalid → <0.3):   {result.low_score_recall:.1%}\n")

    print("Score distribution:")
    for bin_range, count in sorted(result.score_distribution.items()):
        bar = '█' * int(count / result.n_samples * 40)
        pct = count / result.n_samples * 100
        print(f"  {bin_range}: {bar} {count:3d} ({pct:4.1f}%)")

    print(f"\n{'='*60}\n")
