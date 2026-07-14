"""Failure mode taxonomy for benchmark error analysis."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from .benchmark import BenchmarkCaseResult
from .schema import PlannedRoute, RouteRecord


@dataclass(frozen=True)
class FailureMode:
    """Single failure mode instance"""
    category: str  # precursor, operation, condition, validity, modality
    subcategory: str  # specific failure type
    description: str
    target_formula: str
    target_class: str
    modality: str
    severity: str = "moderate"  # low, moderate, high


@dataclass(frozen=True)
class TaxonomyReport:
    """Aggregate failure taxonomy report"""
    total_failures: int
    by_category: dict[str, int] = field(default_factory=dict)
    by_subcategory: dict[str, int] = field(default_factory=dict)
    by_material_class: dict[str, int] = field(default_factory=dict)
    by_modality: dict[str, int] = field(default_factory=dict)
    examples: dict[str, list[dict[str, str]]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


FAILURE_CATEGORIES = {
    "precursor": [
        "wrong_precursor_class",
        "volatile_element_unhandled",
        "missing_element_source",
        "redox_mismatch",
        "precursor_count_mismatch",
    ],
    "operation": [
        "missing_mixing",
        "missing_regrind",
        "missing_wash_dry",
        "insufficient_heating",
        "missing_preparation",
        "wrong_operation_sequence",
    ],
    "condition": [
        "temperature_too_low",
        "temperature_too_high",
        "wrong_atmosphere",
        "insufficient_dwell_time",
        "temperature_mismatch",
    ],
    "validity": [
        "stoichiometry_imbalance",
        "element_coverage_failure",
        "redox_incompatibility",
        "modality_inconsistency",
    ],
}


def analyze_failures(
    cases: list[BenchmarkCaseResult],
    test_routes: dict[str, RouteRecord],
    planned_routes: dict[str, list[PlannedRoute]]
) -> list[FailureMode]:
    """
    Analyze failed benchmark cases and categorize by failure mode.

    Args:
        cases: Benchmark case results
        test_routes: Ground-truth test routes (keyed by route_id)
        planned_routes: Planned routes (keyed by target_formula)

    Returns:
        List of identified failure modes
    """
    failures = []

    for case in cases:
        ground_truth = test_routes.get(case.route_id)
        if not ground_truth:
            continue

        predicted_routes = planned_routes.get(case.target_formula, [])
        if not predicted_routes:
            continue

        predicted = predicted_routes[0]  # Top-1 prediction

        # Analyze different failure types
        if not case.precursor_exact_match and not case.precursor_class_match:
            failure = _diagnose_precursor_failure(case, ground_truth, predicted)
            if failure:
                failures.append(failure)

        if not case.top1_valid:
            failure = _diagnose_validity_failure(case, ground_truth, predicted)
            if failure:
                failures.append(failure)

        if case.operation_similarity < 0.5:
            failure = _diagnose_operation_failure(case, ground_truth, predicted)
            if failure:
                failures.append(failure)

        if case.temperature_error_c and case.temperature_error_c > 100.0:
            failure = _diagnose_condition_failure(case, ground_truth, predicted)
            if failure:
                failures.append(failure)

    return failures


def _diagnose_precursor_failure(
    case: BenchmarkCaseResult,
    ground_truth: RouteRecord,
    predicted: PlannedRoute
) -> FailureMode | None:
    """Diagnose precursor-related failures"""

    gt_classes = set(p.class_name for p in ground_truth.precursors if p.class_name)
    pred_classes = set(p.class_name for p in predicted.precursors if p.class_name)

    # Check for class mismatch
    if "carbonate" in gt_classes and "oxide" in pred_classes and "carbonate" not in pred_classes:
        return FailureMode(
            category="precursor",
            subcategory="wrong_precursor_class",
            description="Used oxide instead of carbonate decomposition route",
            target_formula=case.target_formula,
            target_class=ground_truth.target_class,
            modality=ground_truth.modality,
            severity="moderate",
        )

    if "nitrate" in gt_classes and "oxide" in pred_classes and "nitrate" not in pred_classes:
        return FailureMode(
            category="precursor",
            subcategory="wrong_precursor_class",
            description="Used oxide instead of nitrate decomposition route",
            target_formula=case.target_formula,
            target_class=ground_truth.target_class,
            modality=ground_truth.modality,
            severity="moderate",
        )

    # Check for missing element coverage
    gt_elements = set(ground_truth.target_elements)
    pred_elements = set()
    for p in predicted.precursors:
        pred_elements.update(p.elements)

    if not gt_elements.issubset(pred_elements):
        missing = gt_elements - pred_elements
        return FailureMode(
            category="precursor",
            subcategory="missing_element_source",
            description=f"Missing elements in precursors: {', '.join(missing)}",
            target_formula=case.target_formula,
            target_class=ground_truth.target_class,
            modality=ground_truth.modality,
            severity="high",
        )

    # Check for precursor count mismatch
    if abs(len(ground_truth.precursors) - len(predicted.precursors)) >= 2:
        return FailureMode(
            category="precursor",
            subcategory="precursor_count_mismatch",
            description=f"Precursor count differs significantly (gold: {len(ground_truth.precursors)}, pred: {len(predicted.precursors)})",
            target_formula=case.target_formula,
            target_class=ground_truth.target_class,
            modality=ground_truth.modality,
            severity="low",
        )

    return None


def _diagnose_validity_failure(
    case: BenchmarkCaseResult,
    ground_truth: RouteRecord,
    predicted: PlannedRoute
) -> FailureMode | None:
    """Diagnose validity-related failures"""

    if not predicted.hard_checks.valid:
        # Check what kind of validity failure
        if "element_coverage" in predicted.hard_checks.flags:
            return FailureMode(
                category="validity",
                subcategory="element_coverage_failure",
                description="Route does not cover all target elements",
                target_formula=case.target_formula,
                target_class=ground_truth.target_class,
                modality=ground_truth.modality,
                severity="high",
            )

        if "stoichiometry" in predicted.hard_checks.flags or "balance" in predicted.hard_checks.flags:
            return FailureMode(
                category="validity",
                subcategory="stoichiometry_imbalance",
                description="Route cannot be stoichiometrically balanced",
                target_formula=case.target_formula,
                target_class=ground_truth.target_class,
                modality=ground_truth.modality,
                severity="high",
            )

        if "redox" in predicted.hard_checks.flags:
            return FailureMode(
                category="validity",
                subcategory="redox_incompatibility",
                description="Redox requirements not satisfied by atmosphere/precursors",
                target_formula=case.target_formula,
                target_class=ground_truth.target_class,
                modality=ground_truth.modality,
                severity="moderate",
            )

        if "modality" in predicted.hard_checks.flags:
            return FailureMode(
                category="validity",
                subcategory="modality_inconsistency",
                description="Operations inconsistent with declared modality",
                target_formula=case.target_formula,
                target_class=ground_truth.target_class,
                modality=ground_truth.modality,
                severity="moderate",
            )

    return None


def _diagnose_operation_failure(
    case: BenchmarkCaseResult,
    ground_truth: RouteRecord,
    predicted: PlannedRoute
) -> FailureMode | None:
    """Diagnose operation-related failures"""

    gt_ops = set(op.verb for op in ground_truth.operations)
    pred_ops = set(op.verb for op in predicted.operations)

    # Check for missing critical operations
    if "mix" in gt_ops and "mix" not in pred_ops:
        return FailureMode(
            category="operation",
            subcategory="missing_mixing",
            description="Route lacks initial mixing step present in ground truth",
            target_formula=case.target_formula,
            target_class=ground_truth.target_class,
            modality=ground_truth.modality,
            severity="moderate",
        )

    if ground_truth.modality in {"hydrothermal", "precipitation"}:
        if "wash" in gt_ops and "wash" not in pred_ops:
            return FailureMode(
                category="operation",
                subcategory="missing_wash_dry",
                description="Solution-based route missing wash step",
                target_formula=case.target_formula,
                target_class=ground_truth.target_class,
                modality=ground_truth.modality,
                severity="moderate",
            )

    if ground_truth.modality == "solid_state":
        gt_has_regrind = any("grind" in op.verb.lower() or "mill" in op.verb.lower() for op in ground_truth.operations)
        pred_has_regrind = any("grind" in op.verb.lower() or "mill" in op.verb.lower() for op in predicted.operations)

        if gt_has_regrind and not pred_has_regrind and len(ground_truth.target_elements) >= 3:
            return FailureMode(
                category="operation",
                subcategory="missing_regrind",
                description="Multicomponent solid-state route missing regrinding step",
                target_formula=case.target_formula,
                target_class=ground_truth.target_class,
                modality=ground_truth.modality,
                severity="low",
            )

    return None


def _diagnose_condition_failure(
    case: BenchmarkCaseResult,
    ground_truth: RouteRecord,
    predicted: PlannedRoute
) -> FailureMode | None:
    """Diagnose condition-related failures"""

    if case.temperature_error_c and case.temperature_error_c > 100.0:
        # Determine if too high or too low
        gt_temp = _get_first_heating_temp(ground_truth)
        pred_temp = _get_first_heating_temp(predicted)

        if gt_temp and pred_temp:
            if pred_temp < gt_temp - 100:
                return FailureMode(
                    category="condition",
                    subcategory="temperature_too_low",
                    description=f"Temperature {pred_temp:.0f}°C significantly lower than gold {gt_temp:.0f}°C",
                    target_formula=case.target_formula,
                    target_class=ground_truth.target_class,
                    modality=ground_truth.modality,
                    severity="moderate",
                )
            elif pred_temp > gt_temp + 100:
                return FailureMode(
                    category="condition",
                    subcategory="temperature_too_high",
                    description=f"Temperature {pred_temp:.0f}°C significantly higher than gold {gt_temp:.0f}°C",
                    target_formula=case.target_formula,
                    target_class=ground_truth.target_class,
                    modality=ground_truth.modality,
                    severity="moderate",
                )

    return None


def _get_first_heating_temp(route) -> float | None:
    """Get first heating temperature from route (works for both RouteRecord and PlannedRoute)"""
    for op in route.operations:
        if op.verb == "heat" and hasattr(op, 'temperature_c') and op.temperature_c:
            if hasattr(op.temperature_c, 'midpoint'):
                return op.temperature_c.midpoint
    return None


def generate_taxonomy_report(failures: list[FailureMode]) -> TaxonomyReport:
    """
    Generate aggregate taxonomy report from failure modes.

    Args:
        failures: List of failure modes

    Returns:
        TaxonomyReport with aggregated statistics
    """
    if not failures:
        return TaxonomyReport(total_failures=0)

    by_category = Counter(f.category for f in failures)
    by_subcategory = Counter(f.subcategory for f in failures)
    by_material_class = Counter(f.target_class for f in failures)
    by_modality = Counter(f.modality for f in failures)

    # Collect example cases for each subcategory (up to 3 per subcategory)
    examples = defaultdict(list)
    for failure in failures:
        if len(examples[failure.subcategory]) < 3:
            examples[failure.subcategory].append({
                "target": failure.target_formula,
                "description": failure.description,
                "severity": failure.severity,
            })

    return TaxonomyReport(
        total_failures=len(failures),
        by_category=dict(by_category),
        by_subcategory=dict(by_subcategory),
        by_material_class=dict(by_material_class),
        by_modality=dict(by_modality),
        examples=dict(examples),
    )


def print_taxonomy_report(report: TaxonomyReport) -> None:
    """Print human-readable taxonomy report"""
    print(f"\n{'='*60}")
    print("Failure Mode Taxonomy Report")
    print(f"{'='*60}\n")

    print(f"Total failures analyzed: {report.total_failures}\n")

    if report.total_failures == 0:
        print("No failures to analyze.\n")
        return

    print("By Category:")
    for category, count in sorted(report.by_category.items(), key=lambda x: -x[1]):
        pct = count / report.total_failures * 100
        bar = '█' * int(pct / 2)
        print(f"  {category:15s}: {bar} {count:3d} ({pct:4.1f}%)")

    print("\nBy Subcategory:")
    for subcategory, count in sorted(report.by_subcategory.items(), key=lambda x: -x[1])[:10]:
        pct = count / report.total_failures * 100
        bar = '█' * int(pct / 2)
        print(f"  {subcategory:30s}: {bar} {count:3d} ({pct:4.1f}%)")

    print("\nBy Material Class:")
    for material_class, count in sorted(report.by_material_class.items(), key=lambda x: -x[1]):
        pct = count / report.total_failures * 100
        print(f"  {material_class:15s}: {count:3d} ({pct:4.1f}%)")

    print("\nBy Modality:")
    for modality, count in sorted(report.by_modality.items(), key=lambda x: -x[1]):
        pct = count / report.total_failures * 100
        print(f"  {modality:15s}: {count:3d} ({pct:4.1f}%)")

    print("\nExample Failures:")
    for subcategory, examples in sorted(report.examples.items()):
        if examples:
            print(f"\n  {subcategory}:")
            for ex in examples:
                print(f"    - {ex['target']}: {ex['description']} (severity: {ex['severity']})")

    print(f"\n{'='*60}\n")
