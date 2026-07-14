from synthesis_planner.constraints import evaluate_hard_constraints
from synthesis_planner.formula import infer_target_class, parse_formula
from synthesis_planner.schema import LabConstraints, NumericRange, OperationRecord, PlanningProblem, PlanningState, PrecursorRecord


def _state_for(target_formula="BaTiO3", precursors=(), operations=(), constraints=None):
    problem = PlanningProblem(
        target_formula=target_formula,
        lab_constraints=constraints or LabConstraints(),
    )
    return PlanningState(
        problem=problem,
        target_elements=tuple(sorted(parse_formula(problem.target_formula))),
        target_class=infer_target_class(problem.target_formula),
        stage="terminal",
        precursors=precursors,
        operations=operations,
    )


def test_missing_target_element_is_invalid():
    state = _state_for(
        precursors=(PrecursorRecord("BaCO3", "carbonate", ("Ba", "C", "O")),),
        operations=(OperationRecord("mix", source_label="mix"), OperationRecord("heat", source_label="calcine")),
    )
    result = evaluate_hard_constraints(state)
    assert not result.valid
    assert "missing_target_elements" in result.blocking_flags


def test_temperature_constraint_is_enforced():
    constraints = LabConstraints(max_temperature_c=1000.0)
    state = _state_for(
        precursors=(
            PrecursorRecord("BaCO3", "carbonate", ("Ba", "C", "O")),
            PrecursorRecord("TiO2", "oxide", ("Ti", "O")),
        ),
        operations=(
            OperationRecord("mix", source_label="mix"),
            OperationRecord("heat", temperature_c=NumericRange(1100.0, 1100.0, "C"), source_label="anneal"),
        ),
        constraints=constraints,
    )
    result = evaluate_hard_constraints(state)
    assert not result.valid
    assert "temperature_above_constraint" in result.blocking_flags


def test_solution_only_operation_is_invalid_for_solid_state():
    state = _state_for(
        precursors=(
            PrecursorRecord("BaCO3", "carbonate", ("Ba", "C", "O")),
            PrecursorRecord("TiO2", "oxide", ("Ti", "O")),
        ),
        operations=(
            OperationRecord("mix", source_label="mix"),
            OperationRecord("wash", source_label="wash"),
            OperationRecord("heat", source_label="calcine"),
        ),
    )
    result = evaluate_hard_constraints(state)
    assert not result.valid
    assert "modality_inconsistent_operations" in result.blocking_flags


def test_redox_mismatch_is_blocking():
    state = _state_for(
        target_formula="FeO",
        precursors=(PrecursorRecord("Fe2O3", "oxide", ("Fe", "O")),),
        operations=(
            OperationRecord("mix", source_label="mix"),
            OperationRecord("heat", temperature_c=NumericRange(700.0, 700.0, "C"), atmosphere="air", source_label="anneal"),
        ),
    )
    result = evaluate_hard_constraints(state)
    assert not result.valid
    assert "oxidizing_atmosphere_mismatch" in result.blocking_flags
