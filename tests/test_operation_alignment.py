"""Tests for operation sequence alignment."""

from synthesis_planner.benchmark import _normalize_operation, _operation_similarity
from synthesis_planner.schema import OperationRecord, PlannedRoute, RouteRecord, PrecursorRecord, HardCheckResult, ScoreBreakdown, ThermoAnalysisResult, JudgeResult


def test_normalize_operation():
    assert _normalize_operation("grind") == "grind"
    assert _normalize_operation("mill") == "grind"
    assert _normalize_operation("ball_mill") == "grind"
    assert _normalize_operation("GRIND") == "grind"
    assert _normalize_operation("mix") == "mix"
    assert _normalize_operation("blend") == "mix"
    assert _normalize_operation("calcine") == "calcine"
    assert _normalize_operation("fire") == "calcine"
    assert _normalize_operation("unknown_op") == "unknown_op"


def test_operation_similarity_exact_match():
    """Test perfect match returns 1.0"""
    predicted_route = _make_planned_route(["mix", "grind", "heat", "cool"])
    gold_route = _make_route_record(["mix", "grind", "heat", "cool"])

    similarity = _operation_similarity(predicted_route, gold_route)
    assert similarity == 1.0


def test_operation_similarity_synonym_match():
    """Test that synonyms are recognized"""
    predicted_route = _make_planned_route(["mix", "mill", "calcine", "cool"])
    gold_route = _make_route_record(["mix", "grind", "heat", "cool"])

    similarity = _operation_similarity(predicted_route, gold_route)
    # Should be high since mill->grind and calcine->heat are synonyms
    # SequenceMatcher gives 0.75 for 3/4 matching operations
    assert similarity >= 0.7


def test_operation_similarity_partial_match():
    """Test partial sequence overlap"""
    predicted_route = _make_planned_route(["mix", "grind", "heat"])
    gold_route = _make_route_record(["mix", "grind", "heat", "cool", "regrind", "heat"])

    similarity = _operation_similarity(predicted_route, gold_route)
    # Should be moderate since first 3 operations match but missing latter steps
    assert 0.3 < similarity < 0.8


def test_operation_similarity_no_match():
    """Test completely different sequences"""
    predicted_route = _make_planned_route(["wash", "dry"])
    gold_route = _make_route_record(["mix", "grind", "heat"])

    similarity = _operation_similarity(predicted_route, gold_route)
    assert similarity < 0.3


def test_operation_similarity_empty():
    """Test empty sequences"""
    predicted_route = _make_planned_route([])
    gold_route = _make_route_record([])

    similarity = _operation_similarity(predicted_route, gold_route)
    assert similarity == 1.0


def test_operation_similarity_one_empty():
    """Test one empty sequence"""
    predicted_route = _make_planned_route(["mix", "grind"])
    gold_route = _make_route_record([])

    similarity = _operation_similarity(predicted_route, gold_route)
    assert similarity == 0.0


def test_operation_similarity_order_matters():
    """Test that order affects similarity"""
    predicted_route = _make_planned_route(["heat", "grind", "mix"])
    gold_route = _make_route_record(["mix", "grind", "heat"])

    similarity = _operation_similarity(predicted_route, gold_route)
    # All operations present but wrong order
    assert similarity < 1.0


# Helper functions

def _make_planned_route(verbs: list[str]) -> PlannedRoute:
    operations = tuple(OperationRecord(verb=verb) for verb in verbs)
    return PlannedRoute(
        target_formula="BaTiO3",
        modality="solid_state",
        precursors=(),
        solvents=(),
        operations=operations,
        evidence_dois=(),
        analog_targets=(),
        hard_checks=HardCheckResult(valid=True, flags=(), notes=(), coverage_fraction=1.0, blocking_flags=()),
        score=ScoreBreakdown(
            validity=1.0, stoich=1.0, precursor=1.0, thermo=1.0,
            retrieval=1.0, condition=1.0, llm=1.0, cost=0.0, hazard=0.0, complexity=0.0, total=1.0
        ),
        thermo=ThermoAnalysisResult(
            score=1.0, gas_release_moles=0.0, gas_uptake_moles=0.0,
            byproduct_count=0, decomposition_match=1.0, redox_match=1.0, notes=()
        ),
        judge=JudgeResult(score=1.0, notes=(), flags=()),
        mcts_value=1.0
    )


def _make_route_record(verbs: list[str]) -> RouteRecord:
    operations = tuple(OperationRecord(verb=verb) for verb in verbs)
    return RouteRecord(
        route_id="test",
        source_doi="10.1000/test",
        publication_year=2020,
        modality="solid_state",
        target_formula="BaTiO3",
        target_elements=("Ba", "Ti", "O"),
        chemical_system="Ba-Ti-O",
        target_class="oxide",
        precursors=(),
        solvents=(),
        operations=operations,
        reaction_string="BaCO3 + TiO2 -> BaTiO3 + CO2",
        paragraph_excerpt="Test",
        source_dataset="test"
    )
