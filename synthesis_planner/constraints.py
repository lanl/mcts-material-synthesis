"""Hard route validity checks aligned with proposal-style gating logic."""

from __future__ import annotations

from .formula import parse_formula, safe_required_target_elements
from .schema import HardCheckResult, PlanningState

ALLOWED_VOLATILE_EXTRAS = {"H", "C", "N", "Cl", "Br", "I", "F"}
SOLUTION_ONLY_VERBS = {"wash", "centrifuge", "precipitate", "hydrothermal_hold", "ph_adjust"}
OXIDIZING_ATMOSPHERES = {"air", "oxygen", "o2"}


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

    heating = [operation for operation in state.operations if operation.verb == "heat"]
    if not heating:
        flags.append("missing_heat_step")
        notes.append("Solid-state routes require at least one heat step.")
        blocking.append("missing_heat_step")

    if len(heating) > state.problem.lab_constraints.max_heating_steps:
        flags.append("too_many_heating_steps")
        notes.append("The route exceeds the configured maximum number of heating steps.")
        blocking.append("too_many_heating_steps")

    if state.problem.lab_constraints.require_mixing and not any(operation.verb == "mix" for operation in state.operations):
        flags.append("missing_mixing")
        notes.append("Lab constraints require an explicit mixing or grinding step.")
        blocking.append("missing_mixing")

    allowed_atmospheres = {atm.lower() for atm in state.problem.lab_constraints.allowed_atmospheres}
    for operation in heating:
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
        if allowed_atmospheres and operation.atmosphere:
            parts = {part.strip().lower() for part in operation.atmosphere.split(",") if part.strip()}
            if not parts:
                parts = {operation.atmosphere.lower()}
            if not parts.issubset(allowed_atmospheres):
                flags.append("disallowed_atmosphere")
                notes.append("A heating step uses an atmosphere outside the declared lab constraints.")
                blocking.append("disallowed_atmosphere")

    target_formula_counts = _safe_counts(state.problem.target_formula)
    precursor_counts = {}
    for precursor in state.precursors:
        for element, amount in _safe_counts(precursor.formula).items():
            precursor_counts[element] = precursor_counts.get(element, 0.0) + amount

    non_target_extras = set(precursor_counts) - set(target_formula_counts)
    problematic_extras = non_target_extras - ALLOWED_VOLATILE_EXTRAS
    if problematic_extras:
        flags.append("nonvolatile_extra_elements")
        notes.append(
            "Precursors introduce non-target elements without a supported byproduct assumption: "
            + ",".join(sorted(problematic_extras))
            + "."
        )
        blocking.append("nonvolatile_extra_elements")

    if "O" in target_formula_counts:
        has_oxygen_in_precursors = "O" in precursor_counts
        atmospheres = {operation.atmosphere.lower() for operation in heating if operation.atmosphere}
        if not has_oxygen_in_precursors and not (atmospheres & OXIDIZING_ATMOSPHERES):
            flags.append("missing_oxygen_source")
            notes.append("An oxide target is missing an obvious oxygen source or oxidizing atmosphere.")
            blocking.append("missing_oxygen_source")

    if _target_contains(state.problem.target_formula, {"S", "N"}):
        atmospheres = {operation.atmosphere.lower() for operation in heating if operation.atmosphere}
        if atmospheres & OXIDIZING_ATMOSPHERES:
            flags.append("oxidizing_atmosphere_mismatch")
            notes.append("Oxygen-sensitive target chemistry is paired with an oxidizing atmosphere.")
            blocking.append("oxidizing_atmosphere_mismatch")

    return HardCheckResult(
        valid=not blocking,
        flags=tuple(_dedupe(flags)),
        notes=tuple(_dedupe(notes)),
        coverage_fraction=coverage_fraction,
        blocking_flags=tuple(_dedupe(blocking)),
    )


def _safe_counts(formula: str) -> dict[str, float]:
    try:
        return parse_formula(formula)
    except Exception:
        return {element: 1.0 for element in safe_required_target_elements(formula)} if formula else {}


def _target_contains(formula: str, elements: set[str]) -> bool:
    try:
        counts = parse_formula(formula)
        return any(element in counts for element in elements)
    except Exception:
        return any(element in formula for element in elements)


def _dedupe(items):
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
