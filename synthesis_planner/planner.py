"""High-level planning interface."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

from .datasets import load_processed_routes, prepare_processed_data
from .formula import safe_element_set, safe_infer_target_class
from .mcts import MonteCarloTreeSearch
from .retrieval import RetrievalIndex
from .schema import EvaluationConfig, PlannedRoute, PlanningProblem, PlanningState, RouteRecord
from .scoring import evaluate_state


class SynthesisPlanner:
    def __init__(self, data_dir: str | Path = "data/raw", processed_dir: str | Path = "data/processed", mp_client=None):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.mp_client = mp_client

    def ensure_processed_data(self) -> None:
        solid = self.processed_dir / "solid_state_routes.jsonl"
        solution = self.processed_dir / "solution_routes.jsonl"
        if not solid.exists() or not solution.exists():
            prepare_processed_data(self.data_dir, self.processed_dir)

    def plan(
        self,
        problem: PlanningProblem,
        iterations: int = 250,
        top_k: int = 5,
        exploration_constant: float = 1.4,
        rollout_count: int = 8,
        seed: int | None = None,
        judge_name: str = "deterministic",
        judge_config: dict | None = None,
        use_judge: bool = True,
        use_hard_checks: bool = True,
        use_retrieval: bool = True,
    ) -> list[PlannedRoute]:
        self.ensure_processed_data()
        routes = load_processed_routes(self.processed_dir, problem.modality)
        return self.plan_with_routes(
            problem,
            routes,
            iterations=iterations,
            top_k=top_k,
            exploration_constant=exploration_constant,
            rollout_count=rollout_count,
            seed=seed,
            judge_name=judge_name,
            judge_config=judge_config,
            use_judge=use_judge,
            use_hard_checks=use_hard_checks,
            use_retrieval=use_retrieval,
        )

    def plan_with_routes(
        self,
        problem: PlanningProblem,
        routes,
        iterations: int = 250,
        top_k: int = 5,
        exploration_constant: float = 1.4,
        rollout_count: int = 8,
        seed: int | None = None,
        judge_name: str = "deterministic",
        judge_config: dict | None = None,
        use_judge: bool = True,
        use_hard_checks: bool = True,
        use_retrieval: bool = True,
    ) -> list[PlannedRoute]:
        retrieval = RetrievalIndex(routes)
        analogs = retrieval.retrieve(problem.target_formula, top_k=12) if use_retrieval else []
        candidate_precursor_sets = retrieval.candidate_precursor_sets(problem.target_formula, analogs, max_sets=12)

        root_state = PlanningState(
            problem=problem,
            target_elements=tuple(sorted(safe_element_set(problem.target_formula))),
            target_class=safe_infer_target_class(problem.target_formula),
        )
        mcts = MonteCarloTreeSearch(
            exploration_constant=exploration_constant,
            rollout_count=rollout_count,
            seed=seed,
            evaluation_config=EvaluationConfig(
                judge_name=judge_name,
                use_judge=use_judge,
                use_hard_checks=use_hard_checks,
                judge_config=judge_config or {},
            ),
            mp_client=self.mp_client,
        )
        root = mcts.run(root_state, analogs, candidate_precursor_sets, iterations=iterations)
        return _select_portfolio(root.terminal_routes, top_k)

    def score_route_record(
        self,
        problem: PlanningProblem,
        route: RouteRecord,
        routes: list[RouteRecord],
        judge_name: str = "deterministic",
        judge_config: dict | None = None,
        use_judge: bool = True,
        use_hard_checks: bool = True,
    ) -> PlannedRoute:
        retrieval = RetrievalIndex(routes)
        analogs = retrieval.retrieve(problem.target_formula, top_k=12)
        state = PlanningState(
            problem=problem,
            target_elements=tuple(sorted(safe_element_set(problem.target_formula))),
            target_class=safe_infer_target_class(problem.target_formula),
            stage="terminal",
            precursors=route.precursors,
            solvents=route.solvents,
            operations=route.operations,
            evidence_dois=tuple(r.source_doi for _, r in analogs[:5] if r.source_doi),
            analog_targets=tuple(r.target_formula for _, r in analogs[:5]),
        )
        return evaluate_state(
            state,
            analogs,
            EvaluationConfig(
                judge_name=judge_name,
                use_judge=use_judge,
                use_hard_checks=use_hard_checks,
                judge_config=judge_config or {},
            ),
        )

    def plan_nearest_neighbor(
        self,
        problem: PlanningProblem,
        routes: list[RouteRecord],
        top_k: int = 5,
        judge_name: str = "deterministic",
        judge_config: dict | None = None,
        use_judge: bool = True,
        use_hard_checks: bool = True,
    ) -> list[PlannedRoute]:
        retrieval = RetrievalIndex(routes)
        analogs = retrieval.retrieve(problem.target_formula, top_k=max(top_k, 12))
        planned = [
            self.score_route_record(
                problem,
                route,
                routes,
                judge_name=judge_name,
                judge_config=judge_config,
                use_judge=use_judge,
                use_hard_checks=use_hard_checks,
            )
            for _, route in analogs[:top_k]
        ]
        return _select_portfolio(planned, top_k)

    def plan_frequency_prior(
        self,
        problem: PlanningProblem,
        routes: list[RouteRecord],
        top_k: int = 5,
        judge_name: str = "deterministic",
        judge_config: dict | None = None,
        use_judge: bool = True,
        use_hard_checks: bool = True,
    ) -> list[PlannedRoute]:
        retrieval = RetrievalIndex(routes)
        analogs = retrieval.retrieve(problem.target_formula, top_k=12)
        candidate_precursor_sets = retrieval.candidate_precursor_sets(problem.target_formula, [], max_sets=top_k)
        planned = []
        for _, precursor_set in candidate_precursor_sets:
            current = PlanningState(
                problem=problem,
                target_elements=tuple(sorted(safe_element_set(problem.target_formula))),
                target_class=safe_infer_target_class(problem.target_formula),
                stage="precursors",
            )
            from .grammar import apply_action, expand_state
            from .schema import Action

            current = apply_action(current, Action("set_precursors", "precursors", 1.0, precursor_set), analogs)
            while not current.is_terminal:
                actions = expand_state(current, analogs, candidate_precursor_sets)
                if not actions:
                    break
                current = apply_action(current, max(actions, key=lambda action: action.prior), analogs)
            planned.append(
                evaluate_state(
                    current,
                    analogs,
                    EvaluationConfig(
                        judge_name=judge_name,
                        use_judge=use_judge,
                        use_hard_checks=use_hard_checks,
                        judge_config=judge_config or {},
                    ),
                )
            )
        return _select_portfolio(planned, top_k)

    def save_routes(self, routes: list[PlannedRoute], output_dir: str | Path) -> Path:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        target = routes[0].target_formula if routes else "empty"
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        path = out_dir / f"{target}_{timestamp}.json"
        with path.open("w") as handle:
            json.dump([route.to_dict() for route in routes], handle, indent=2)
        return path


def _select_portfolio(routes: list[PlannedRoute], top_k: int, diversity_threshold: float = 0.7) -> list[PlannedRoute]:
    """
    Select top-k diverse routes using greedy diversity selection.

    Args:
        routes: List of candidate routes
        top_k: Number of routes to select
        diversity_threshold: Routes with similarity > threshold are penalized

    Returns:
        List of diverse high-scoring routes
    """
    if not routes:
        return []

    # Sort by score
    ranked = sorted(routes, key=lambda route: route.score.total, reverse=True)

    # Greedily select diverse routes
    portfolio = [ranked[0]]  # Start with highest-scoring route

    for route in ranked[1:]:
        if len(portfolio) >= top_k:
            break

        # Compute similarity to existing portfolio
        max_similarity = max(_route_similarity(route, p) for p in portfolio)

        # Accept if sufficiently different from portfolio
        if max_similarity < diversity_threshold:
            portfolio.append(route)
        else:
            # Near-duplicate: only accept if score is significantly better than current portfolio
            min_portfolio_score = min(p.score.total for p in portfolio)
            if route.score.total > min_portfolio_score * 1.15:  # 15% better
                portfolio.append(route)

    return portfolio[:top_k]


def _route_similarity(route1: PlannedRoute, route2: PlannedRoute) -> float:
    """
    Compute similarity between two routes [0, 1].

    Based on:
    - Precursor formula overlap (Jaccard)
    - Operation sequence overlap (Jaccard)
    - Temperature proximity
    """
    # Precursor similarity (Jaccard on formulas)
    precursors1 = set(p.formula for p in route1.precursors)
    precursors2 = set(p.formula for p in route2.precursors)

    if precursors1 or precursors2:
        precursor_sim = len(precursors1 & precursors2) / len(precursors1 | precursors2)
    else:
        precursor_sim = 1.0

    # Operation similarity (Jaccard on verbs)
    ops1 = set(op.verb for op in route1.operations)
    ops2 = set(op.verb for op in route2.operations)

    if ops1 or ops2:
        operation_sim = len(ops1 & ops2) / len(ops1 | ops2)
    else:
        operation_sim = 1.0

    # Temperature similarity
    temp1 = _get_first_heating_temp(route1)
    temp2 = _get_first_heating_temp(route2)

    if temp1 is not None and temp2 is not None:
        temp_diff = abs(temp1 - temp2)
        # Normalize: 100°C difference = 0.5 similarity, 0°C = 1.0
        temp_sim = max(0.0, 1.0 - temp_diff / 200.0)
    else:
        temp_sim = 0.5  # Unknown, moderate similarity

    # Weighted average
    return 0.4 * precursor_sim + 0.4 * operation_sim + 0.2 * temp_sim


def _get_first_heating_temp(route: PlannedRoute) -> float | None:
    """Extract first heating temperature from route operations"""
    for op in route.operations:
        if op.verb == "heat" and op.temperature_c and op.temperature_c.midpoint is not None:
            return op.temperature_c.midpoint
    return None
