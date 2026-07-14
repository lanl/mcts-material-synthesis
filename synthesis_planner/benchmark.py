"""Benchmark split generation, baselines, and retrospective evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import json
from pathlib import Path
import random
from statistics import mean
from typing import Callable

from .datasets import load_processed_routes
from .planner import SynthesisPlanner
from .schema import PlannedRoute, PlanningProblem, RouteRecord


# Operation synonym mapping for normalization
OPERATION_SYNONYMS = {
    "calcine": ["calcine", "fire"],
    "sinter": ["sinter", "anneal"],
    "grind": ["grind", "mill", "ball_mill", "crush", "pestle"],
    "mix": ["mix", "blend", "combine", "stir"],
    "wash": ["wash", "rinse"],
    "dry": ["dry"],
    "cool": ["cool", "quench", "air_cool"],
    "heat": ["heat", "calcine", "fire"],
    "precipitate": ["precipitate", "precipitation"],
    "pelletize": ["pelletize", "press", "compact"],
    "dissolve": ["dissolve", "add_solvent"],
    "filter": ["filter", "filtrate"],
    "centrifuge": ["centrifuge", "separate"],
}


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
    analog_support_score: float = 0.0
    closest_analog_formula: str = ""
    closest_analog_similarity: float = 0.0


@dataclass(frozen=True)
class BenchmarkSummary:
    method: str
    split_type: str
    n_train: int
    n_test: int
    precursor_exact_match_at_1: float
    precursor_class_match_at_1: float
    top1_validity_rate: float
    mean_operation_similarity: float
    mean_temperature_error_c: float | None
    mean_analog_support: float = 0.0
    cases: list[BenchmarkCaseResult] = None

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "split_type": self.split_type,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "precursor_exact_match_at_1": self.precursor_exact_match_at_1,
            "precursor_class_match_at_1": self.precursor_class_match_at_1,
            "top1_validity_rate": self.top1_validity_rate,
            "mean_operation_similarity": self.mean_operation_similarity,
            "mean_temperature_error_c": self.mean_temperature_error_c,
            "mean_analog_support": self.mean_analog_support,
            "cases": [asdict(case) for case in self.cases] if self.cases else [],
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
    method: str = "mcts",
    modality: str = "solid_state",
    iterations: int = 50,
    top_k: int = 1,
    rollout_count: int = 3,
    seed: int | None = None,
    judge_name: str = "deterministic",
    judge_config: dict | None = None,
) -> BenchmarkSummary:
    from .retrieval import RetrievalIndex

    retrieval = RetrievalIndex(train_routes)
    cases = []
    for index, gold in enumerate(test_routes):
        predictions = _predict_with_method(
            method,
            planner,
            PlanningProblem(target_formula=gold.target_formula, modality=modality),
            train_routes,
            iterations=iterations,
            top_k=top_k,
            rollout_count=rollout_count,
            seed=(seed or 0) + index,
            judge_name=judge_name,
            judge_config=judge_config,
        )
        if not predictions:
            continue
        top = predictions[0]

        # Retrieve analogs for this target
        analogs = retrieval.retrieve(gold.target_formula, top_k=12)
        analog_support, closest_formula, closest_sim = _compute_analog_support(top, analogs)

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
                analog_support_score=analog_support,
                closest_analog_formula=closest_formula,
                closest_analog_similarity=closest_sim,
            )
        )

    if not cases:
        return BenchmarkSummary(method, split_type, len(train_routes), len(test_routes), 0.0, 0.0, 0.0, 0.0, None, 0.0, [])

    temp_errors = [case.temperature_error_c for case in cases if case.temperature_error_c is not None]
    return BenchmarkSummary(
        method=method,
        split_type=split_type,
        n_train=len(train_routes),
        n_test=len(test_routes),
        precursor_exact_match_at_1=mean(1.0 if case.precursor_exact_match else 0.0 for case in cases),
        precursor_class_match_at_1=mean(1.0 if case.precursor_class_match else 0.0 for case in cases),
        top1_validity_rate=mean(1.0 if case.top1_valid else 0.0 for case in cases),
        mean_operation_similarity=mean(case.operation_similarity for case in cases),
        mean_temperature_error_c=mean(temp_errors) if temp_errors else None,
        mean_analog_support=mean(case.analog_support_score for case in cases),
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


def load_routes(processed_dir: str | Path, modality: str) -> list[RouteRecord]:
    return load_processed_routes(processed_dir, modality)


def evaluate_method_suite(
    planner: SynthesisPlanner,
    train_routes: list[RouteRecord],
    test_routes: list[RouteRecord],
    split_type: str,
    methods: list[str],
    modality: str = "solid_state",
    iterations: int = 50,
    top_k: int = 1,
    rollout_count: int = 3,
    seed: int | None = None,
    judge_name: str = "deterministic",
    judge_config: dict | None = None,
) -> dict[str, dict]:
    return {
        method: evaluate_split(
            planner,
            train_routes,
            test_routes,
            split_type=split_type,
            method=method,
            modality=modality,
            iterations=iterations,
            top_k=top_k,
            rollout_count=rollout_count,
            seed=seed,
            judge_name=judge_name,
            judge_config=judge_config,
        ).to_dict()
        for method in methods
    }


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
    """
    Compute operation sequence similarity using edit distance.
    Returns [0, 1] where 1 is perfect match.
    """
    predicted_verbs = [operation.verb for operation in predicted.operations]
    gold_verbs = [operation.verb for operation in gold.operations if operation.verb != "start"]

    if not predicted_verbs and not gold_verbs:
        return 1.0
    if not predicted_verbs or not gold_verbs:
        return 0.0

    # Normalize operation names to canonical forms
    pred_normalized = [_normalize_operation(verb) for verb in predicted_verbs]
    gold_normalized = [_normalize_operation(verb) for verb in gold_verbs]

    # Compute sequence similarity using SequenceMatcher (longest common subsequence based)
    matcher = SequenceMatcher(None, pred_normalized, gold_normalized)
    return matcher.ratio()


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


def _normalize_operation(verb: str) -> str:
    """
    Map operation verb to canonical name using synonym dictionary.
    Returns original verb if no mapping found.
    """
    verb_lower = verb.lower().strip()
    for canonical, synonyms in OPERATION_SYNONYMS.items():
        if verb_lower in synonyms:
            return canonical
    return verb_lower


def _compute_analog_support(
    route: PlannedRoute,
    analogs: list[tuple[float, RouteRecord]]
) -> tuple[float, str, float]:
    """
    Measure how well the route is supported by retrieved analogs.
    Returns (support_score, closest_analog_formula, closest_analog_similarity).

    Support score is based on:
    - Precursor class overlap with analogs
    - Operation sequence overlap with analogs
    - Weighted by retrieval similarity
    """
    if not analogs:
        return 0.0, "", 0.0

    closest_analog_formula = analogs[0][1].target_formula
    closest_analog_similarity = analogs[0][0]

    # Find most similar analog route (considering both retrieval and route overlap)
    best_support = 0.0
    for retrieval_similarity, analog in analogs[:5]:  # Check top 5 analogs
        # Precursor class overlap
        route_precursor_classes = set(p.class_name for p in route.precursors if p.class_name)
        analog_precursor_classes = set(p.class_name for p in analog.precursors if p.class_name)

        if route_precursor_classes and analog_precursor_classes:
            precursor_overlap = len(route_precursor_classes & analog_precursor_classes) / len(route_precursor_classes | analog_precursor_classes)
        else:
            precursor_overlap = 0.0

        # Operation overlap
        route_ops = set(_normalize_operation(op.verb) for op in route.operations)
        analog_ops = set(_normalize_operation(op.verb) for op in analog.operations)

        if route_ops and analog_ops:
            operation_overlap = len(route_ops & analog_ops) / len(route_ops | analog_ops)
        else:
            operation_overlap = 0.0

        # Combined support: weight retrieval similarity with route overlap
        support = retrieval_similarity * (0.5 * precursor_overlap + 0.5 * operation_overlap)
        best_support = max(best_support, support)

    return best_support, closest_analog_formula, closest_analog_similarity


def _predict_with_method(
    method: str,
    planner: SynthesisPlanner,
    problem: PlanningProblem,
    train_routes: list[RouteRecord],
    iterations: int,
    top_k: int,
    rollout_count: int,
    seed: int,
    judge_name: str,
    judge_config: dict | None,
) -> list[PlannedRoute]:
    if method == "mcts":
        return planner.plan_with_routes(
            problem,
            train_routes,
            iterations=iterations,
            top_k=top_k,
            rollout_count=rollout_count,
            seed=seed,
            judge_name=judge_name,
            judge_config=judge_config,
        )
    if method == "nearest_neighbor":
        return planner.plan_nearest_neighbor(problem, train_routes, top_k=top_k, judge_name=judge_name, judge_config=judge_config)
    if method == "frequency_prior":
        return planner.plan_frequency_prior(problem, train_routes, top_k=top_k, judge_name=judge_name, judge_config=judge_config)
    if method == "mcts_no_retrieval":
        return planner.plan_with_routes(
            problem,
            train_routes,
            iterations=iterations,
            top_k=top_k,
            rollout_count=rollout_count,
            seed=seed,
            judge_name=judge_name,
            judge_config=judge_config,
            use_retrieval=False,
        )
    if method == "mcts_no_judge":
        return planner.plan_with_routes(problem, train_routes, iterations=iterations, top_k=top_k, rollout_count=rollout_count, seed=seed, use_judge=False, judge_name="none")
    if method == "mcts_no_hard_checks":
        return planner.plan_with_routes(
            problem,
            train_routes,
            iterations=iterations,
            top_k=top_k,
            rollout_count=rollout_count,
            seed=seed,
            judge_name=judge_name,
            judge_config=judge_config,
            use_hard_checks=False,
        )
    raise ValueError(f"Unknown benchmark method: {method}")
