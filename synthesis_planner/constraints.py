"""Hard route validity checks aligned with proposal-style gating logic."""

from __future__ import annotations

from .chemistry import analyze_redox, balance_route
from .formula import safe_required_target_elements
from .schema import HardCheckResult, PlanningState

SOLUTION_ONLY_VERBS = {"wash", "centrifuge", "precipitate", "hydrothermal_hold", "ph_adjust", "age"}


def evaluate_hard_constraints(state: PlanningState) -> HardCheckResult:
    flags = []
    notes = []
    blocking = []

    required = safe_required_target_elements(state.problem.target_formula)
    coverage = {element for precursor in state.precursors for element in precursor.elements}
    coverage_fraction = len(required & coverage) / max(1, len(required))

    if required - coverage:
        missing = ",".join(sorted(required - coverage))
        flags.append("missing_target_elements")
        notes.append(f"Missing target element coverage: {missing}.")
        blocking.append("missing_target_elements")

    if len(state.precursors) > state.problem.lab_constraints.max_precursors:
        flags.append("too_many_precursors")
        notes.append("The route exceeds the allowed precursor count.")
        blocking.append("too_many_precursors")

    if any(precursor.class_name in state.problem.lab_constraints.forbidden_precursor_classes for precursor in state.precursors):
        flags.append("forbidden_precursor_class")
        notes.append("The route uses a precursor class forbidden by lab constraints.")
        blocking.append("forbidden_precursor_class")

    if state.problem.modality == "solid_state":
        solution_verbs = {operation.verb for operation in state.operations if operation.verb in SOLUTION_ONLY_VERBS}
        if solution_verbs:
            flags.append("modality_inconsistent_operations")
            notes.append("Solution-only operations appeared in a solid-state route.")
            blocking.append("modality_inconsistent_operations")

    thermal_operations = [
        operation for operation in state.operations if operation.verb in {"heat", "hydrothermal_hold", "calcine", "anneal"}
    ]
    if state.problem.modality in {"solid_state", "hydrothermal"} and not thermal_operations:
        flags.append("missing_heat_step")
        notes.append(f"{state.problem.modality.replace('_', ' ').title()} routes require at least one heat step.")
        blocking.append("missing_heat_step")

    if len(thermal_operations) > state.problem.lab_constraints.max_heating_steps:
        flags.append("too_many_heating_steps")
        notes.append("The route exceeds the configured maximum number of heating steps.")
        blocking.append("too_many_heating_steps")

    if state.problem.lab_constraints.require_mixing and not any(operation.verb == "mix" for operation in state.operations):
        flags.append("missing_mixing")
        notes.append("Lab constraints require an explicit mixing or grinding step.")
        blocking.append("missing_mixing")

    if state.problem.modality in {"hydrothermal", "precipitation"} and not state.solvents:
        flags.append("missing_solvent")
        notes.append("Solution-phase routes require an explicit solvent choice.")
        blocking.append("missing_solvent")

    allowed_atmospheres = {atm.lower() for atm in state.problem.lab_constraints.allowed_atmospheres}
    for operation in thermal_operations:
        if operation.temperature_c and operation.temperature_c.midpoint is not None:
            temp = operation.temperature_c.midpoint
            if state.problem.lab_constraints.min_temperature_c is not None and temp < state.problem.lab_constraints.min_temperature_c:
                flags.append("temperature_below_constraint")
                notes.append("A heating step is below the allowed temperature range.")
                blocking.append("temperature_below_constraint")
            if state.problem.lab_constraints.max_temperature_c is not None and temp > state.problem.lab_constraints.max_temperature_c:
                flags.append("temperature_above_constraint")
                notes.append("A heating step exceeds the allowed temperature range.")
                blocking.append("temperature_above_constraint")
            if state.problem.modality == "hydrothermal" and temp > 350.0:
                flags.append("hydrothermal_temperature_too_high")
                notes.append("Hydrothermal temperatures above 350 C are implausible for standard autoclave routes.")
                blocking.append("hydrothermal_temperature_too_high")
        if allowed_atmospheres and operation.atmosphere:
            parts = {part.strip().lower() for part in operation.atmosphere.split(",") if part.strip()}
            if not parts:
                parts = {operation.atmosphere.lower()}
            if not parts.issubset(allowed_atmospheres):
                flags.append("disallowed_atmosphere")
                notes.append("A heating step uses an atmosphere outside the declared lab constraints.")
                blocking.append("disallowed_atmosphere")

    balance = balance_route(state)
    if not balance.feasible:
        flags.append("stoichiometric_imbalance")
        notes.append("The precursor set could not be balanced to the target with common volatile products/reactants.")
        blocking.append("stoichiometric_imbalance")
        if balance.residual_elements:
            residual = ", ".join(f"{element}:{amount:.3g}" for element, amount in sorted(balance.residual_elements.items()))
            notes.append(f"Residual unmatched elements remain after balancing: {residual}.")
    if balance.unused_precursors:
        flags.append("unused_precursor")
        notes.append("At least one selected precursor does not participate in the balanced reaction stoichiometry.")

    redox = analyze_redox(state, balance)
    if redox.flags:
        flags.extend(redox.flags)
        notes.extend(redox.notes)
    for flag in redox.flags:
        if flag in {"missing_oxidant", "missing_reductant", "oxidizing_atmosphere_mismatch", "reducing_atmosphere_mismatch"}:
            blocking.append(flag)

    if state.problem.modality == "precipitation":
        verbs = {operation.verb for operation in state.operations}
        if "precipitate" not in verbs:
            flags.append("missing_precipitation_step")
            notes.append("Precipitation routes require an explicit precipitation step.")
            blocking.append("missing_precipitation_step")
        if "wash" not in verbs or "dry" not in verbs:
            flags.append("incomplete_postprocessing")
            notes.append("Precipitation routes generally require wash and dry post-processing.")
            blocking.append("incomplete_postprocessing")

    return HardCheckResult(
        valid=not blocking,
        flags=tuple(_dedupe(flags)),
        notes=tuple(_dedupe(notes)),
        coverage_fraction=coverage_fraction,
        blocking_flags=tuple(_dedupe(blocking)),
        reaction_balance=balance,
        redox=redox,
    )


def _dedupe(items):
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
