"""Hard constraints, heuristics, and deterministic judge logic."""

from __future__ import annotations

from statistics import mean, median

from .chemistry import analyze_thermodynamics
from .constraints import evaluate_hard_constraints
from .formula import safe_required_target_elements
from .judge import build_judge
from .physics import reaction_enthalpy, thermo_favorability
from .schema import EvaluationConfig, PlannedRoute, PlanningState, PrecursorRecord, RouteRecord, ScoreBreakdown


def evaluate_state(state: PlanningState, analogs: list[tuple[float, RouteRecord]], config: EvaluationConfig | None = None, mp_client=None) -> PlannedRoute:
    config = config or EvaluationConfig()
    hard_checks = evaluate_hard_constraints(state)
    thermo_analysis = analyze_thermodynamics(state, hard_checks.reaction_balance, hard_checks.redox, mp_client=mp_client)

    stoich = _stoich_score(hard_checks)
    validity = 1.0 if hard_checks.valid or not config.use_hard_checks else 0.0
    retrieval = max((score for score, _ in analogs), default=0.0) / 10.0
    precursor = _precursor_score(state.precursors, state.problem.target_formula, config)
    condition = _condition_score(state, analogs)
    thermo = thermo_analysis.score
    judge = build_judge(
        config.judge_name if config.use_judge else "none",
        config.judge_config,
    ).evaluate(state, analogs, hard_checks)
    complexity = len(state.operations) / 10.0
    cost = max(0.0, (len(state.precursors) - 2) * 0.1)
    hazard = _hazard_score(state)
    stock_score = _stock_score(state.precursors, config)
    # Physics signal: thermodynamic favorability of the balanced reaction. A
    # non-analog signal (computed from formation enthalpies, not retrieval) that
    # is available even for novel targets via the oxide-sum estimator.
    delta_h, computable = reaction_enthalpy(
        state, hard_checks.reaction_balance, getattr(config, "physics_provider", None)
    )
    physics = thermo_favorability(delta_h, computable)

    total = (
        (1.5 * validity if config.use_hard_checks else 0.0)
        + 1.2 * stoich
        + 1.0 * precursor
        + 0.6 * thermo
        # Retrieval self-similarity is sharply demoted: it should *guide*
        # expansion via priors, not reward the leaf (fixes the top1_validity~1.0
        # circularity from EVALUATION.md). Default weight 0.2 vs the old 1.0.
        + config.retrieval_weight * retrieval
        # Condition quality is weighted more heavily (0.8 -> 1.3): since MCTS
        # inherits the frequency prior on precursors, conditions/stages are where
        # search must add value over that prior.
        + 1.3 * condition
        + (1.0 * judge.score if config.use_judge else 0.0)
        # Solved-to-stock is the dominant terminal reward (analog of "a route to
        # purchasable stock was found"). Only credited on terminal states with a
        # concrete precursor set and a stock context supplied via the config.
        + config.stock_weight * stock_score
        # Physics favorability, mean-centered so neutral (0.5) adds nothing and a
        # thermodynamically downhill reaction is rewarded / uphill penalized.
        + config.physics_weight * (physics - 0.5)
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
            physics=physics,
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


def _condition_score(state: PlanningState, analogs: list | None = None) -> float:
    """Graded, data-driven condition quality in [0,1].

    Redesigned (Fix #3) so the score does NOT saturate: the dominant term is a
    *continuous* temperature-appropriateness signal measured against the analog
    temperature distribution (falling back to physical bands when analogs are
    weak). This gives MCTS a real gradient to optimize conditions over the
    frequency prior's static defaults, and makes a missing calcination genuinely
    costly for precipitation-derived crystalline oxides.
    """
    ops = state.operations
    modality = state.problem.modality
    heating = [op for op in ops if op.verb == "heat"]
    # Key step must be present for the modality.
    if modality in {"solid_state", "hydrothermal"} and not heating:
        return 0.0
    if modality == "precipitation" and not any(op.verb == "precipitate" for op in ops):
        return 0.0

    temperatures = [op.temperature_c.midpoint for op in heating if op.temperature_c and op.temperature_c.midpoint is not None]
    analog_temps = _analog_heat_temperatures(analogs)

    # Process-completeness (minor, non-saturating) component.
    score = 0.0
    if any(op.verb == "mix" for op in ops):
        score += 0.1
    if modality in {"precipitation", "hydrothermal"}:
        if state.solvents:
            score += 0.1
        if any(op.verb == "wash" for op in ops) and any(op.verb == "dry" for op in ops):
            score += 0.1

    # Temperature-appropriateness (dominant, continuous) component.
    if modality == "solid_state":
        score += 0.55 * _temp_appropriateness(temperatures, analog_temps, (650.0, 1250.0))
    elif modality == "hydrothermal":
        score += 0.55 * _temp_appropriateness(temperatures, analog_temps, (100.0, 250.0))
    else:  # precipitation: a calcination is required to crystallize the oxide.
        score += 0.15  # precipitate present (guarded above)
        score += 0.45 * _temp_appropriateness(temperatures, analog_temps, (300.0, 900.0))
    return min(score, 1.0)


def _analog_heat_temperatures(analogs: list | None) -> list[float]:
    if not analogs:
        return []
    temps = []
    for _, route in analogs:
        for op in route.operations:
            if op.verb == "heat" and op.temperature_c and op.temperature_c.midpoint is not None:
                temps.append(op.temperature_c.midpoint)
    return temps


def _temp_appropriateness(temperatures: list[float], analog_temps: list[float], band: tuple[float, float]) -> float:
    """Continuous [0,1] credit for how appropriate the heating temperature is.

    Prefers proximity to the analog median temperature; falls back to a physical
    band with graded fall-off. Returns 0 when no heating temperature is present
    (so a missing/untempered firing earns no temperature credit).
    """
    if not temperatures:
        return 0.0
    avg_temp = mean(temperatures)
    if analog_temps:
        target = median(analog_temps)
        return max(0.0, 1.0 - abs(avg_temp - target) / 300.0)
    lo, hi = band
    if lo <= avg_temp <= hi:
        return 1.0
    dist = (lo - avg_temp) if avg_temp < lo else (avg_temp - hi)
    return max(0.0, 1.0 - dist / 300.0)


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
