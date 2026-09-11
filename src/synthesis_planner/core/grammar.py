"""Modality-aware synthesis grammar for MCTS expansion."""

from __future__ import annotations

from collections import Counter
from statistics import median

from .schema import Action, OperationRecord, PlanningState, PrecursorRecord, RouteRecord

# Elements treated as anion / non-metal framework when counting target cations
# for template-family lookup. Kept local to avoid a core->data import (mirrors
# data.stock.ANION_ELEMENTS).
_ANION_ELEMENTS = frozenset({"O", "H", "C", "N", "S", "P", "F", "Cl", "Br", "I", "Se", "Te", "B"})


def _cation_count(elements: tuple[str, ...]) -> int:
    return len([el for el in elements if el not in _ANION_ELEMENTS])


def expand_state(
    state: PlanningState,
    analogs: list[tuple[float, RouteRecord]],
    candidate_precursor_sets: list[tuple[float, tuple[PrecursorRecord, ...]]],
    templates=None,
) -> list[Action]:
    if state.problem.modality in {"hydrothermal", "precipitation"}:
        return _expand_solution_state(state, analogs, candidate_precursor_sets)

    if state.stage == "precursors":
        return [
            Action(
                kind="set_precursors",
                label=", ".join(precursor.formula for precursor in precursors),
                prior=max(0.1, float(score)),
                payload=precursors,
            )
            for score, precursors in candidate_precursor_sets
        ]

    if state.stage == "preparation":
        return [
            Action("set_preparation", "mix -> grind", 0.9, _prep_ops("grind")),
            Action("set_preparation", "mix -> ball_mill", 0.6, _prep_ops("ball_mill")),
            Action("set_preparation", "mix -> grind -> pelletize", 0.55, _prep_ops("grind", include_shape=True)),
        ]

    if state.stage == "heating":
        return _heating_actions(analogs, state, templates)

    if state.stage == "finalize":
        return [
            Action("finalize", "terminate", 1.0, ()),
            Action("finalize", "slow cool", 0.45, (OperationRecord(verb="cool", source_label="slow cool"),)),
            Action("finalize", "quench", 0.35, (OperationRecord(verb="quench", source_label="quench"),)),
        ]

    return []


def apply_action(state: PlanningState, action: Action, analogs: list[tuple[float, RouteRecord]]) -> PlanningState:
    if state.problem.modality in {"hydrothermal", "precipitation"}:
        return _apply_solution_action(state, action, analogs)

    if action.kind == "set_precursors":
        top_dois = tuple(route.source_doi for _, route in analogs[:5] if route.source_doi)
        top_targets = tuple(route.target_formula for _, route in analogs[:5])
        return PlanningState(
            problem=state.problem,
            target_elements=state.target_elements,
            target_class=state.target_class,
            stage="preparation",
            precursors=tuple(action.payload),
            solvents=state.solvents,
            operations=state.operations,
            evidence_dois=top_dois,
            analog_targets=top_targets,
        )

    if action.kind == "set_preparation":
        return PlanningState(
            problem=state.problem,
            target_elements=state.target_elements,
            target_class=state.target_class,
            stage="heating",
            precursors=state.precursors,
            solvents=state.solvents,
            operations=state.operations + tuple(action.payload),
            evidence_dois=state.evidence_dois,
            analog_targets=state.analog_targets,
        )

    if action.kind == "add_heat_stage":
        # Append one firing stage and remain in the heating MDP so the search can
        # decide whether to add another stage or stop.
        return PlanningState(
            problem=state.problem,
            target_elements=state.target_elements,
            target_class=state.target_class,
            stage="heating",
            precursors=state.precursors,
            solvents=state.solvents,
            operations=state.operations + tuple(action.payload),
            evidence_dois=state.evidence_dois,
            analog_targets=state.analog_targets,
        )

    if action.kind in {"set_heating", "finish_heating"}:
        # ``set_heating`` = full-schedule shortcut; ``finish_heating`` = stop the
        # multi-stage MDP. Both advance to finalize.
        return PlanningState(
            problem=state.problem,
            target_elements=state.target_elements,
            target_class=state.target_class,
            stage="finalize",
            precursors=state.precursors,
            solvents=state.solvents,
            operations=state.operations + tuple(action.payload),
            evidence_dois=state.evidence_dois,
            analog_targets=state.analog_targets,
        )

    if action.kind == "finalize":
        return PlanningState(
            problem=state.problem,
            target_elements=state.target_elements,
            target_class=state.target_class,
            stage="terminal",
            precursors=state.precursors,
            solvents=state.solvents,
            operations=state.operations + tuple(action.payload),
            evidence_dois=state.evidence_dois,
            analog_targets=state.analog_targets,
        )

    raise ValueError(f"Unknown action kind: {action.kind}")


def rollout_completion(state: PlanningState, analogs: list[tuple[float, RouteRecord]], candidate_precursor_sets: list[tuple[float, tuple[PrecursorRecord, ...]]], rng, templates=None) -> PlanningState:
    current = state
    while not current.is_terminal:
        actions = expand_state(current, analogs, candidate_precursor_sets, templates)
        if not actions:
            break
        total_prior = sum(action.prior for action in actions)
        threshold = rng.random() * total_prior
        cumulative = 0.0
        chosen = actions[-1]
        for action in actions:
            cumulative += action.prior
            if cumulative >= threshold:
                chosen = action
                break
        current = apply_action(current, chosen, analogs)
    return current


def _prep_ops(grinding_label: str, include_shape: bool = False) -> tuple[OperationRecord, ...]:
    ops = [
        OperationRecord(verb="mix", source_label="mix"),
        OperationRecord(verb=grinding_label, source_label=grinding_label.replace("_", " ")),
    ]
    if include_shape:
        ops.append(OperationRecord(verb="shape", source_label="pelletize"))
    return tuple(ops)


def _heating_actions(analogs: list[tuple[float, RouteRecord]], state: PlanningState | None = None, templates=None) -> list[Action]:
    temperatures = []
    durations = []
    atmospheres = Counter()
    multi_step_examples = []
    for _, route in analogs:
        heating_ops = [operation for operation in route.operations if operation.verb == "heat"]
        if heating_ops:
            for op in heating_ops:
                if op.temperature_c and op.temperature_c.midpoint is not None:
                    temperatures.append(op.temperature_c.midpoint)
                if op.time_h and op.time_h.midpoint is not None:
                    durations.append(op.time_h.midpoint)
                if op.atmosphere:
                    atmospheres[op.atmosphere] += 1
            if len(heating_ops) >= 2:
                multi_step_examples.append(tuple(heating_ops[:2]))

    median_temp = round(median(temperatures), 1) if temperatures else 900.0
    median_time = round(median(durations), 1) if durations else 8.0
    atmosphere = atmospheres.most_common(1)[0][0] if atmospheres else "air"

    # Fold in the mined per-family condition schedule (WS-B) when available: it
    # backfills a data-grounded temperature spread + atmosphere even where the
    # analog set is sparse, widening the heating action space for the search.
    schedule = None
    if templates is not None and state is not None:
        try:
            schedule = templates.condition_schedule(state.target_class, _cation_count(state.target_elements))
        except Exception:
            schedule = None
    if schedule is not None:
        for temp in (schedule.temperature_p25, schedule.temperature_p50, schedule.temperature_p75):
            if temp is not None:
                temperatures.append(temp)
        if schedule.temperature_p50 is not None and not atmospheres:
            median_temp = schedule.temperature_p50
        if schedule.dwell_h_median and not durations:
            median_time = schedule.dwell_h_median
        if schedule.top_atmosphere and not atmospheres:
            atmosphere = schedule.top_atmosphere

    # Multi-stage firing MDP (#1): heating is a *variable-length* sequential
    # decision, not a fixed preset bundle. At each heating node the search may
    # either append one more firing stage (``add_heat_stage``) or stop
    # (``finish_heating``). Route length - how many calcine/regrind/anneal
    # stages - is therefore itself searched, mirroring real solid-state practice
    # (single calcine; calcine -> regrind -> re-fire; etc.). A hard stage cap
    # bounds the tree.
    n_stages = sum(1 for op in state.operations if op.verb == "heat") if state is not None else 0

    # Once the cap is reached the only legal move is to stop.
    if n_stages >= _MAX_HEAT_STAGES:
        return [Action("finish_heating", "done heating", 1.0, ())]

    actions: list[Action] = []
    temp_options = _temperature_setpoints(temperatures, median_temp)
    for label, temp, prior in temp_options:
        heat_op = OperationRecord(
            verb="heat",
            temperature_c=_range(temp),
            time_h=_range(median_time, units="h"),
            atmosphere=atmosphere,
            source_label="calcine" if n_stages == 0 else "anneal",
        )
        if n_stages == 0:
            # First firing: a bare calcine stage.
            actions.append(Action("add_heat_stage", f"calcine ({label})", prior, (heat_op,)))
        else:
            # Subsequent firings are preceded by an intergrind, as is standard
            # for completing solid-state reactions / improving homogeneity.
            actions.append(
                Action(
                    "add_heat_stage",
                    f"regrind + re-fire ({label})",
                    prior * 0.8,
                    (OperationRecord(verb="mix", source_label="regrind"), heat_op),
                )
            )

    # Stopping is only legal after at least one firing stage (a solid-state route
    # with no heating is rejected downstream anyway).
    if n_stages >= 1:
        actions.append(Action("finish_heating", "done heating", 1.0, ()))

    # Data-grounded shortcut: a full literature multistep schedule reachable in
    # one move (kept so known good sequences stay directly selectable).
    if n_stages == 0 and multi_step_examples:
        actions.append(Action("set_heating", "literature-style multistep", 0.95, multi_step_examples[0]))
    return actions


# Maximum number of firing stages the search may stack (depth cap for the
# multi-stage heating MDP). >3-stage solid-state schedules are vanishingly rare.
_MAX_HEAT_STAGES = 3


def _temperature_setpoints(temperatures: list[float], median_temp: float) -> list[tuple[str, float, float]]:
    """Return (label, temperature_C, prior) single-step heating options.

    Uses the 25th/50th/75th percentiles of the observed analog temperatures so
    the search can trade off between cooler and hotter data-supported setpoints.
    Falls back to a single median action when there is too little data to form a
    meaningful spread.
    """
    median_action = ("single heat step", median_temp, 1.0)
    distinct = sorted(set(round(t, 1) for t in temperatures))
    if len(distinct) < 4:
        return [median_action]

    ordered = sorted(temperatures)

    def _percentile(p: float) -> float:
        idx = min(len(ordered) - 1, max(0, int(round(p * (len(ordered) - 1)))))
        return round(ordered[idx], 1)

    low = _percentile(0.25)
    high = _percentile(0.75)
    options = [median_action]
    # Only add tails that are meaningfully separated from the median (>=40 C).
    if median_temp - low >= 40.0:
        options.append(("cooler heat step", low, 0.7))
    if high - median_temp >= 40.0:
        options.append(("hotter heat step", high, 0.7))
    return options


def _range(midpoint: float, units: str = "C"):
    from .schema import NumericRange

    return NumericRange(midpoint, midpoint, units)


def _expand_solution_state(state: PlanningState, analogs: list[tuple[float, RouteRecord]], candidate_precursor_sets: list[tuple[float, tuple[PrecursorRecord, ...]]]) -> list[Action]:
    if state.stage == "precursors":
        return [
            Action(
                kind="set_precursors",
                label=", ".join(precursor.formula for precursor in precursors),
                prior=max(0.1, float(score)),
                payload=precursors,
            )
            for score, precursors in candidate_precursor_sets
        ]
    if state.stage == "solution_setup":
        return _solution_setup_actions(state.problem.modality, analogs)
    if state.stage == "reaction":
        return _solution_reaction_actions(state.problem.modality, analogs)
    if state.stage == "postprocess":
        return _solution_postprocess_actions(state.problem.modality, analogs, state)
    if state.stage == "finalize":
        return [
            Action("finalize", "terminate", 1.0, ()),
            Action("finalize", "cool -> terminate", 0.4, (OperationRecord(verb="cool", source_label="cool"),)),
        ]
    return []


def _apply_solution_action(state: PlanningState, action: Action, analogs: list[tuple[float, RouteRecord]]) -> PlanningState:
    if action.kind == "set_precursors":
        top_dois = tuple(route.source_doi for _, route in analogs[:5] if route.source_doi)
        top_targets = tuple(route.target_formula for _, route in analogs[:5])
        return PlanningState(
            problem=state.problem,
            target_elements=state.target_elements,
            target_class=state.target_class,
            stage="solution_setup",
            precursors=tuple(action.payload),
            solvents=state.solvents,
            operations=state.operations,
            evidence_dois=top_dois,
            analog_targets=top_targets,
        )
    if action.kind == "set_solution_setup":
        payload = action.payload
        return PlanningState(
            problem=state.problem,
            target_elements=state.target_elements,
            target_class=state.target_class,
            stage="reaction",
            precursors=state.precursors,
            solvents=tuple(payload["solvents"]),
            operations=state.operations + tuple(payload["operations"]),
            evidence_dois=state.evidence_dois,
            analog_targets=state.analog_targets,
        )
    if action.kind == "set_solution_reaction":
        return PlanningState(
            problem=state.problem,
            target_elements=state.target_elements,
            target_class=state.target_class,
            stage="postprocess",
            precursors=state.precursors,
            solvents=state.solvents,
            operations=state.operations + tuple(action.payload),
            evidence_dois=state.evidence_dois,
            analog_targets=state.analog_targets,
        )
    if action.kind == "set_solution_postprocess":
        return PlanningState(
            problem=state.problem,
            target_elements=state.target_elements,
            target_class=state.target_class,
            stage="finalize",
            precursors=state.precursors,
            solvents=state.solvents,
            operations=state.operations + tuple(action.payload),
            evidence_dois=state.evidence_dois,
            analog_targets=state.analog_targets,
        )
    if action.kind == "finalize":
        return PlanningState(
            problem=state.problem,
            target_elements=state.target_elements,
            target_class=state.target_class,
            stage="terminal",
            precursors=state.precursors,
            solvents=state.solvents,
            operations=state.operations + tuple(action.payload),
            evidence_dois=state.evidence_dois,
            analog_targets=state.analog_targets,
        )
    raise ValueError(f"Unknown action kind: {action.kind}")


def _solution_setup_actions(modality: str, analogs: list[tuple[float, RouteRecord]]) -> list[Action]:
    solvent_counter = Counter()
    for _, route in analogs:
        for solvent in route.solvents:
            solvent_counter[solvent.lower()] += 1
    common = [name for name, _ in solvent_counter.most_common(3)]
    if not common:
        common = ["water", "ethanol", "water,ethanol"]
    actions = []
    for idx, solvent in enumerate(common):
        actions.append(
            Action(
                kind="set_solution_setup",
                label=f"{solvent} solution",
                prior=max(0.4, 1.0 - 0.15 * idx),
                payload={
                    "solvents": tuple(part.strip() for part in solvent.split(",") if part.strip()),
                    "operations": (OperationRecord(verb="mix", source_label=f"dissolve in {solvent}"),),
                },
            )
        )
    return actions


def _solution_reaction_actions(modality: str, analogs: list[tuple[float, RouteRecord]]) -> list[Action]:
    heating_temps = []
    heating_times = []
    for _, route in analogs:
        for operation in route.operations:
            if operation.verb == "heat":
                if operation.temperature_c and operation.temperature_c.midpoint is not None:
                    heating_temps.append(operation.temperature_c.midpoint)
                if operation.time_h and operation.time_h.midpoint is not None:
                    heating_times.append(operation.time_h.midpoint)

    if modality == "hydrothermal":
        temp = round(median(heating_temps), 1) if heating_temps else 180.0
        dwell = round(median(heating_times), 1) if heating_times else 12.0
        return [
            Action(
                kind="set_solution_reaction",
                label="hydrothermal hold",
                prior=1.0,
                payload=(
                    OperationRecord(
                        verb="heat",
                        temperature_c=_range(min(max(temp, 100.0), 250.0)),
                        time_h=_range(min(max(dwell, 4.0), 48.0), units="h"),
                        atmosphere="sealed",
                        source_label="hydrothermal hold",
                    ),
                ),
            )
        ]

    return [
        Action(
            kind="set_solution_reaction",
            label="precipitate and age",
            prior=1.0,
            payload=(
                OperationRecord(verb="precipitate", source_label="precipitate"),
                OperationRecord(verb="age", time_h=_range(2.0, units="h"), source_label="age"),
            ),
        ),
        Action(
            kind="set_solution_reaction",
            label="precipitate only",
            prior=0.7,
            payload=(OperationRecord(verb="precipitate", source_label="precipitate"),),
        ),
    ]


# Target classes that are crystalline phases and therefore essentially always
# require a calcination of the as-precipitated/dried gel to form the product.
_CALCINATION_REQUIRED_CLASSES = frozenset({"oxide", "phosphate", "nitride", "sulfide", "halide", "other"})


def _solution_postprocess_actions(
    modality: str,
    analogs: list[tuple[float, RouteRecord]],
    state: PlanningState | None = None,
) -> list[Action]:
    wash_dry = (
        OperationRecord(verb="wash", source_label="wash"),
        OperationRecord(verb="dry", source_label="dry"),
    )
    if modality == "hydrothermal":
        # A hydrothermal hold often crystallises the product directly, so a final
        # anneal is optional (lower prior).
        return [
            Action("set_solution_postprocess", "wash -> dry", 1.0, wash_dry),
            Action(
                "set_solution_postprocess",
                "wash -> dry -> anneal",
                0.7,
                wash_dry
                + (
                    OperationRecord(
                        verb="heat",
                        temperature_c=_range(400.0),
                        time_h=_range(4.0, units="h"),
                        source_label="post-anneal",
                    ),
                ),
            ),
        ]

    # Precipitation / sol-gel: the dried solid is an amorphous hydroxide / oxalate
    # / gel, NOT the crystalline target. Calcination is required to form the
    # product, so it is the dominant action; a bare wash->dry (target already
    # crystalline as-precipitated) is the low-prior exception.
    needs_calcination = state is None or state.target_class in _CALCINATION_REQUIRED_CLASSES
    calcine = Action(
        "set_solution_postprocess",
        "wash -> dry -> calcine",
        1.0 if needs_calcination else 0.75,
        wash_dry
        + (
            OperationRecord(
                verb="heat",
                temperature_c=_range(500.0),
                time_h=_range(3.0, units="h"),
                source_label="calcine",
            ),
        ),
    )
    wash_dry_only = Action(
        "set_solution_postprocess",
        "wash -> dry",
        0.5 if needs_calcination else 1.0,
        wash_dry,
    )
    return [calcine, wash_dry_only] if needs_calcination else [wash_dry_only, calcine]
