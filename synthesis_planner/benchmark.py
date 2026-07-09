"""Benchmark split generation and retrospective evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from statistics import mean
from typing import Callable

from .datasets import load_processed_routes
from .formula import required_target_elements
from .planner import SynthesisPlanner
from .schema import PlannedRoute, PlanningProblem, RouteRecord


@dataclass(frozen=True)
class BenchmarkCaseResult:
    route_id: str
    target_formula: str
    split_key: str
    precursor_exact_match: bool
    precursor_class_match: bool
    top1_valid: bool
    operation_similarity: float
    temperature_error_c: float | None


@dataclass(frozen=True)
class BenchmarkSummary:
    split_type: str
    n_train: int
    n_test: int
    precursor_exact_match_at_1: float
    precursor_class_match_at_1: float
    top1_validity_rate: float
    mean_operation_similarity: float
    mean_temperature_error_c: float | None
    cases: list[BenchmarkCaseResult]

    def to_dict(self) -> dict:
        return {
            "split_type": self.split_type,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "precursor_exact_match_at_1": self.precursor_exact_match_at_1,
            "precursor_class_match_at_1": self.precursor_class_match_at_1,
            "top1_validity_rate": self.top1_validity_rate,
            "mean_operation_similarity": self.mean_operation_similarity,
            "mean_temperature_error_c": self.mean_temperature_error_c,
            "cases": [asdict(case) for case in self.cases],
        }


def build_split(routes: list[RouteRecord], split_type: str, test_fraction: float = 0.2, seed: int | None = None) -> tuple[list[RouteRecord], list[RouteRecord]]:
    rng = random.Random(seed)

    if split_type == "random":
        shuffled = routes[:]
        rng.shuffle(shuffled)
        cutoff = max(1, int(len(shuffled) * (1.0 - test_fraction)))
        return shuffled[:cutoff], shuffled[cutoff:]

    key_fn = _split_key_fn(split_type)
    grouped = {}
    for route in routes:
        key = key_fn(route)
        if key is None:
            continue
        grouped.setdefault(key, []).append(route)

    keys = list(grouped)
    if split_type == "publication_year":
        keys.sort()
    else:
        rng.shuffle(keys)

    if not keys:
        return routes, []

    n_test_keys = max(1, int(len(keys) * test_fraction))
    if split_type == "publication_year":
        test_keys = set(keys[-n_test_keys:])
    else:
        test_keys = set(keys[:n_test_keys])

    train, test = [], []
    for key, members in grouped.items():
        (test if key in test_keys else train).extend(members)
    return train, test


def evaluate_split(
    planner: SynthesisPlanner,
    train_routes: list[RouteRecord],
    test_routes: list[RouteRecord],
    split_type: str,
    iterations: int = 50,
    top_k: int = 1,
    rollout_count: int = 3,
    seed: int | None = None,
) -> BenchmarkSummary:
    cases = []
    for index, gold in enumerate(test_routes):
        predictions = planner.plan_with_routes(
            PlanningProblem(target_formula=gold.target_formula, modality="solid_state"),
            train_routes,
            iterations=iterations,
            top_k=top_k,
            rollout_count=rollout_count,
            seed=(seed or 0) + index,
        )
        if not predictions:
            continue
        top = predictions[0]
        cases.append(
            BenchmarkCaseResult(
                route_id=gold.route_id,
                target_formula=gold.target_formula,
                split_key=str(_split_key_fn(split_type)(gold)),
                precursor_exact_match=_precursor_exact_match(top, gold),
                precursor_class_match=_precursor_class_match(top, gold),
                top1_valid=top.hard_checks.valid,
                operation_similarity=_operation_similarity(top, gold),
                temperature_error_c=_temperature_error(top, gold),
            )
        )

    if not cases:
        return BenchmarkSummary(split_type, len(train_routes), len(test_routes), 0.0, 0.0, 0.0, 0.0, None, [])

    temp_errors = [case.temperature_error_c for case in cases if case.temperature_error_c is not None]
    return BenchmarkSummary(
        split_type=split_type,
        n_train=len(train_routes),
        n_test=len(test_routes),
        precursor_exact_match_at_1=mean(1.0 if case.precursor_exact_match else 0.0 for case in cases),
        precursor_class_match_at_1=mean(1.0 if case.precursor_class_match else 0.0 for case in cases),
        top1_validity_rate=mean(1.0 if case.top1_valid else 0.0 for case in cases),
        mean_operation_similarity=mean(case.operation_similarity for case in cases),
        mean_temperature_error_c=mean(temp_errors) if temp_errors else None,
        cases=cases,
    )


def save_benchmark_summary(summary: BenchmarkSummary, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(summary.to_dict(), handle, indent=2)
    return output


def save_split_manifest(train_routes: list[RouteRecord], test_routes: list[RouteRecord], split_type: str, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split_type": split_type,
        "train_route_ids": [route.route_id for route in train_routes],
        "test_route_ids": [route.route_id for route in test_routes],
    }
    with output.open("w") as handle:
        json.dump(payload, handle, indent=2)
    return output


def load_solid_state_routes(processed_dir: str | Path) -> list[RouteRecord]:
    return load_processed_routes(processed_dir, "solid_state")


def _split_key_fn(split_type: str) -> Callable[[RouteRecord], str | int | None]:
    if split_type == "target_formula":
        return lambda route: route.target_formula
    if split_type == "chemical_system":
        return lambda route: route.chemical_system
    if split_type == "material_family":
        return lambda route: route.target_class
    if split_type == "publication_year":
        return lambda route: route.publication_year
    if split_type == "random":
        return lambda route: route.route_id
    raise ValueError(f"Unknown split type: {split_type}")


def _precursor_exact_match(predicted: PlannedRoute, gold: RouteRecord) -> bool:
    return {precursor.formula for precursor in predicted.precursors} == {precursor.formula for precursor in gold.precursors}


def _precursor_class_match(predicted: PlannedRoute, gold: RouteRecord) -> bool:
    return {precursor.class_name for precursor in predicted.precursors} == {precursor.class_name for precursor in gold.precursors}


def _operation_similarity(predicted: PlannedRoute, gold: RouteRecord) -> float:
    predicted_verbs = [operation.verb for operation in predicted.operations]
    gold_verbs = [operation.verb for operation in gold.operations if operation.verb != "start"]
    if not predicted_verbs and not gold_verbs:
        return 1.0
    if not predicted_verbs or not gold_verbs:
        return 0.0

    pred_counts = Counter(predicted_verbs)
    gold_counts = Counter(gold_verbs)
    overlap = sum((pred_counts & gold_counts).values())
    precision = overlap / len(predicted_verbs)
    recall = overlap / len(gold_verbs)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _temperature_error(predicted: PlannedRoute, gold: RouteRecord) -> float | None:
    pred = _first_heating_temperature(predicted.operations)
    gold_temp = _first_heating_temperature(gold.operations)
    if pred is None or gold_temp is None:
        return None
    return abs(pred - gold_temp)


def _first_heating_temperature(operations) -> float | None:
    for operation in operations:
        if operation.verb == "heat" and operation.temperature_c and operation.temperature_c.midpoint is not None:
            return operation.temperature_c.midpoint
    return None
