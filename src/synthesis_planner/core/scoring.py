"""Hard constraints, heuristics, and deterministic judge logic."""

from __future__ import annotations

from statistics import mean

from .chemistry import analyze_thermodynamics
from .constraints import evaluate_hard_constraints
from .formula import safe_required_target_elements
from .judge import build_judge
from .schema import EvaluationConfig, PlannedRoute, PlanningState, PrecursorRecord, RouteRecord, ScoreBreakdown


def evaluate_state(state: PlanningState, analogs: list[tuple[float, RouteRecord]], config: EvaluationConfig | None = None, mp_client=None) -> PlannedRoute:
    config = config or EvaluationConfig()
    hard_checks = evaluate_hard_constraints(state)
    thermo_analysis = analyze_thermodynamics(state, hard_checks.reaction_balance, hard_checks.redox, mp_client=mp_client)

    stoich = _stoich_score(hard_checks)
    validity = 1.0 if hard_checks.valid or not config.use_hard_checks else 0.0
    retrieval = max((score for score, _ in analogs), default=0.0) / 10.0
    precursor = _precursor_score(state.precursors, state.problem.target_formula, config)
    condition = _condition_score(state)
    thermo = thermo_analysis.score
    judge = build_judge(
        config.judge_name if config.use_judge else "none",
        config.judge_config,
    ).evaluate(state, analogs, hard_checks)
    complexity = len(state.operations) / 10.0
    cost = max(0.0, (len(state.precursors) - 2) * 0.1)
    hazard = _hazard_score(state)
    stock_score = _stock_score(state.precursors, config)

    total = (
        (1.5 * validity if config.use_hard_checks else 0.0)
        + 1.2 * stoich
        + 1.0 * precursor
        + 0.6 * thermo
        # Retrieval self-similarity is sharply demoted: it should *guide*
        # expansion via priors, not reward the leaf (fixes the top1_validity~1.0
        # circularity from EVALUATION.md). Default weight 0.2 vs the old 1.0.
        + config.retrieval_weight * retrieval
        + 0.8 * condition
        + (1.0 * judge.score if config.use_judge else 0.0)
        # Solved-to-stock is the dominant terminal reward (analog of "a route to
        # purchasable stock was found"). Only credited on terminal states with a
        # concrete precursor set and a stock context supplied via the config.
        + config.stock_weight * stock_score
        - 0.4 * cost
        - 0.5 * hazard
        - 0.3 * complexity
    )
    if config.use_hard_checks and not hard_checks.valid:
        total -= 2.0 + 0.25 * len(hard_checks.blocking_flags)

    return PlannedRoute(
        target_formula=state.problem.target_formula,
        modality=state.problem.modality,
        precursors=state.precursors,
        solvents=state.solvents,
        operations=state.operations,
        evidence_dois=state.evidence_dois,
        analog_targets=state.analog_targets,
        hard_checks=hard_checks,
        score=ScoreBreakdown(
            validity=validity,
            stoich=stoich,
            precursor=precursor,
            thermo=thermo,
            retrieval=retrieval,
            condition=condition,
            llm=judge.score,
            cost=cost,
            hazard=hazard,
            complexity=complexity,
            total=total,
            stock=stock_score,
        ),
        thermo=thermo_analysis,
        judge=judge,
        mcts_value=total,
    )


def _stock_score(precursors: tuple[PrecursorRecord, ...], config: EvaluationConfig) -> float:
    """Fraction of leaf precursors in stock (0.0 when no stock context / no precursors).

    ``config.stock`` is duck-typed: any object exposing ``contains(precursor)``.
    This is the solved-to-stock signal that dominates the terminal reward and
    de-circularizes success (reaching buildable stock, not agreeing with
    retrieval). A fully-in-stock route scores 1.0.
    """
    stock = getattr(config, "stock", None)
    if stock is None or not precursors:
        return 0.0
    in_stock = sum(1 for precursor in precursors if stock.contains(precursor))
    return in_stock / len(precursors)


def _precursor_score(
    precursors: tuple[PrecursorRecord, ...],
    target_formula: str,
    config: EvaluationConfig | None = None,
) -> float:
    classes = {precursor.class_name for precursor in precursors}
    target_lower = target_formula.lower()
    score = 0.4
    if "oxide" in classes:
        score += 0.2
    if "carbonate" in classes or "nitrate" in classes:
        score += 0.1
    if "sulfide" in target_lower and "sulfide" in classes:
        score += 0.2
    if "nitride" in target_lower and "halide" not in classes:
        score += 0.05
    class_score = min(score, 1.0)

    # Blend in the data-frequency prior when available: a route built from the
    # corpus's most common precursors scores higher, aligning MCTS's precursor
    # preference with a frequency-prior baseline (see EvaluationConfig).
    freq_score = _precursor_frequency_score(precursors, config)
    if freq_score is None:
        return class_score
    return min(1.0, 0.5 * class_score + 0.5 * freq_score)


def _precursor_frequency_score(
    precursors: tuple[PrecursorRecord, ...], config: EvaluationConfig | None
) -> float | None:
    """Mean normalized corpus frequency of the chosen precursor formulas, or None.

    ``config.precursor_frequency`` is a duck-typed mapping formula -> [0,1].
    """
    fmap = getattr(config, "precursor_frequency", None) if config is not None else None
    if not fmap or not precursors:
        return None
    return sum(float(fmap.get(p.formula, 0.0)) for p in precursors) / len(precursors)


def _condition_score(state: PlanningState) -> float:
    heating = [operation for operation in state.operations if operation.verb == "heat"]
    if state.problem.modality in {"solid_state", "hydrothermal"} and not heating:
        return 0.0
    if state.problem.modality == "precipitation" and not any(operation.verb == "precipitate" for operation in state.operations):
        return 0.0
    temperatures = [op.temperature_c.midpoint for op in heating if op.temperature_c and op.temperature_c.midpoint is not None]
    score = 0.5
    if state.problem.modality == "solid_state" and temperatures:
        avg_temp = mean(temperatures)
        if 650.0 <= avg_temp <= 1250.0:
            score += 0.25
        if state.target_class == "oxide" and avg_temp >= 750.0:
            score += 0.15
    if state.problem.modality == "hydrothermal" and temperatures:
        avg_temp = mean(temperatures)
        if 100.0 <= avg_temp <= 250.0:
            score += 0.3
        if state.solvents:
            score += 0.1
        if any(operation.verb == "wash" for operation in state.operations) and any(operation.verb == "dry" for operation in state.operations):
            score += 0.1
    if state.problem.modality == "precipitation":
        if any(operation.verb == "precipitate" for operation in state.operations):
            score += 0.2
        if state.solvents:
            score += 0.1
        if any(operation.verb == "wash" for operation in state.operations) and any(operation.verb == "dry" for operation in state.operations):
            score += 0.15
    if any(operation.verb == "mix" for operation in state.operations):
        score += 0.1
    return min(score, 1.0)


def _stoich_score(hard_checks) -> float:
    score = 0.4 * hard_checks.coverage_fraction
    if hard_checks.reaction_balance and hard_checks.reaction_balance.feasible:
        score += 0.5
        if not hard_checks.reaction_balance.unused_precursors:
            score += 0.1
    return min(score, 1.0)


def _hazard_score(state: PlanningState) -> float:
    score = 0.0
    formulas = [precursor.formula for precursor in state.precursors]
    if any("NH4" in formula or "NO3" in formula for formula in formulas):
        score += 0.15
    if any("Cl" in formula or "Br" in formula for formula in formulas):
        score += 0.1
    if state.target_class == "sulfide":
        score += 0.2
    return min(score, 1.0)
