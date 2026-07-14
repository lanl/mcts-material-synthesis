"""Tests for partial-state judging."""

from synthesis_planner.judge import DeterministicJudge
from synthesis_planner.schema import (
    OperationRecord,
    PlanningProblem,
    PlanningState,
    PrecursorRecord,
    RouteRecord,
)


def test_partial_judge_precursors_stage():
    """Test partial judging after precursor selection"""
    judge = DeterministicJudge()

    # State with precursors but missing element
    state = PlanningState(
        problem=PlanningProblem(target_formula="BaTiO3"),
        target_elements=("Ba", "Ti", "O"),
        target_class="oxide",
        stage="precursors",
        precursors=(
            # Missing Ba source
            PrecursorRecord(formula="TiO2", class_name="oxide", elements=("Ti", "O")),
        ),
    )

    analogs = []
    result = judge.evaluate_partial(state, analogs)

    # Should flag missing element
    assert "missing_element_source" in result.flags
    assert len(result.notes) > 0
    assert result.score < 0.5  # Low score for missing elements


def test_partial_judge_precursors_complete():
    """Test partial judging with complete precursor coverage"""
    judge = DeterministicJudge()

    state = PlanningState(
        problem=PlanningProblem(target_formula="BaTiO3"),
        target_elements=("Ba", "Ti", "O"),
        target_class="oxide",
        stage="precursors",
        precursors=(
            PrecursorRecord(formula="BaCO3", class_name="carbonate", elements=("Ba", "C", "O")),
            PrecursorRecord(formula="TiO2", class_name="oxide", elements=("Ti", "O")),
        ),
    )

    analogs = []
    result = judge.evaluate_partial(state, analogs)

    # Should not flag missing elements
    assert "missing_element_source" not in result.flags
    assert result.score >= 0.5


def test_partial_judge_preparation_missing_mixing():
    """Test partial judging for missing mixing step"""
    judge = DeterministicJudge()

    state = PlanningState(
        problem=PlanningProblem(target_formula="BaTiO3"),
        target_elements=("Ba", "Ti", "O"),
        target_class="oxide",
        stage="preparation",
        precursors=(
            PrecursorRecord(formula="BaCO3", class_name="carbonate", elements=("Ba", "C", "O")),
            PrecursorRecord(formula="TiO2", class_name="oxide", elements=("Ti", "O")),
        ),
        operations=(),  # No mixing
    )

    analogs = []
    result = judge.evaluate_partial(state, analogs)

    # Should flag missing mixing
    assert "missing_mixing" in result.flags
    assert result.score < 0.5


def test_partial_judge_heating_regrind_need():
    """Test partial judging for regrinding needs"""
    judge = DeterministicJudge()

    # Use a target with 3+ cation elements to trigger regrind suggestion
    state = PlanningState(
        problem=PlanningProblem(target_formula="YBa2Cu3O7", modality="solid_state"),
        target_elements=("Y", "Ba", "Cu", "O"),
        target_class="oxide",
        stage="heating",
        precursors=(
            PrecursorRecord(formula="Y2O3", class_name="oxide", elements=("Y", "O")),
            PrecursorRecord(formula="BaCO3", class_name="carbonate", elements=("Ba", "C", "O")),
            PrecursorRecord(formula="CuO", class_name="oxide", elements=("Cu", "O")),
        ),
        operations=(
            OperationRecord(verb="mix"),
            OperationRecord(verb="heat"),
        ),
    )

    analogs = []
    result = judge.evaluate_partial(state, analogs)

    # Should suggest regrinding for multicomponent system (3+ cations)
    assert "potential_regrind_need" in result.flags
    assert len(result.notes) > 0


def test_partial_judge_finalize_hydrothermal():
    """Test partial judging for hydrothermal post-processing"""
    judge = DeterministicJudge()

    state = PlanningState(
        problem=PlanningProblem(target_formula="CoFe2O4", modality="hydrothermal"),
        target_elements=("Co", "Fe", "O"),
        target_class="oxide",
        stage="finalize",
        precursors=(
            PrecursorRecord(formula="Co(NO3)2", class_name="nitrate", elements=("Co", "N", "O")),
            PrecursorRecord(formula="Fe(NO3)3", class_name="nitrate", elements=("Fe", "N", "O")),
        ),
        operations=(
            OperationRecord(verb="mix"),
            OperationRecord(verb="heat"),
            # Missing wash and dry
        ),
    )

    analogs = []
    result = judge.evaluate_partial(state, analogs)

    # Should flag incomplete post-processing
    assert "incomplete_postprocessing" in result.flags
    assert result.score < 0.7


def test_partial_judge_all_good():
    """Test partial judging when route looks good at current stage"""
    judge = DeterministicJudge()

    state = PlanningState(
        problem=PlanningProblem(target_formula="BaTiO3"),
        target_elements=("Ba", "Ti", "O"),
        target_class="oxide",
        stage="precursors",
        precursors=(
            PrecursorRecord(formula="BaCO3", class_name="carbonate", elements=("Ba", "C", "O")),
            PrecursorRecord(formula="TiO2", class_name="oxide", elements=("Ti", "O")),
        ),
    )

    analogs = []
    result = judge.evaluate_partial(state, analogs)

    # Should have no serious flags
    assert len([f for f in result.flags if "missing" in f]) == 0
    assert result.score >= 0.5


def test_partial_judge_base_class():
    """Test that base judge returns neutral result for partial evaluation"""
    from synthesis_planner.judge import BaseJudge

    judge = BaseJudge()
    state = PlanningState(
        problem=PlanningProblem(target_formula="BaTiO3"),
        target_elements=("Ba", "Ti", "O"),
        target_class="oxide",
        stage="precursors",
        precursors=(),
    )

    analogs = []
    result = judge.evaluate_partial(state, analogs)

    # Base class should return neutral result
    assert result.score == 0.5
    assert result.uncertainty == 0.5
