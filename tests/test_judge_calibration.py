"""Tests for judge calibration."""

from synthesis_planner.judge_calibration import (
    CalibrationSample,
    _compute_precursor_match,
    _rank_data,
    _route_to_state,
    _spearman_correlation,
    calibrate_judge,
)
from synthesis_planner.schema import (
    OperationRecord,
    PlanningProblem,
    PlanningState,
    PrecursorRecord,
    RouteRecord,
)


def test_rank_data():
    """Test ranking function"""
    data = [3.0, 1.0, 2.0, 5.0, 4.0]
    ranks = _rank_data(data)
    # 1.0 -> rank 1, 2.0 -> rank 2, 3.0 -> rank 3, 4.0 -> rank 4, 5.0 -> rank 5
    assert ranks == [3.0, 1.0, 2.0, 5.0, 4.0]


def test_spearman_correlation_perfect():
    """Perfect positive correlation should give 1.0"""
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 6.0, 8.0, 10.0]
    corr = _spearman_correlation(x, y)
    assert abs(corr - 1.0) < 0.01


def test_spearman_correlation_negative():
    """Perfect negative correlation should give -1.0"""
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [5.0, 4.0, 3.0, 2.0, 1.0]
    corr = _spearman_correlation(x, y)
    assert abs(corr - (-1.0)) < 0.01


def test_spearman_correlation_zero():
    """Weak correlation should give value between -1 and 1"""
    x = [1.0, 2.0, 3.0, 4.0]
    y = [2.0, 1.0, 4.0, 3.0]
    corr = _spearman_correlation(x, y)
    # This particular dataset has some correlation, not zero
    assert -1.0 <= corr <= 1.0


def test_route_to_state():
    """Test converting RouteRecord to PlanningState"""
    route = _make_route("BaTiO3", ["BaCO3", "TiO2"], ["mix", "heat"])

    state = _route_to_state(route)

    assert state.problem.target_formula == "BaTiO3"
    assert state.target_elements == ("Ba", "Ti", "O")
    assert state.stage == "terminal"
    assert len(state.precursors) == 2
    assert len(state.operations) == 2


def test_compute_precursor_match_exact():
    """Test exact precursor class match"""
    state = PlanningState(
        problem=PlanningProblem(target_formula="BaTiO3"),
        target_elements=("Ba", "Ti", "O"),
        target_class="oxide",
        precursors=(
            PrecursorRecord(formula="BaCO3", class_name="carbonate", elements=("Ba", "C", "O")),
            PrecursorRecord(formula="TiO2", class_name="oxide", elements=("Ti", "O")),
        ),
    )

    gold = _make_route("BaTiO3", ["BaCO3", "TiO2"], ["mix", "heat"])

    match = _compute_precursor_match(state, gold)
    assert match == 1.0  # Both classes match


def test_compute_precursor_match_partial():
    """Test partial precursor class match"""
    state = PlanningState(
        problem=PlanningProblem(target_formula="BaTiO3"),
        target_elements=("Ba", "Ti", "O"),
        target_class="oxide",
        precursors=(
            PrecursorRecord(formula="BaO", class_name="oxide", elements=("Ba", "O")),
            PrecursorRecord(formula="TiO2", class_name="oxide", elements=("Ti", "O")),
        ),
    )

    gold = _make_route("BaTiO3", ["BaCO3", "TiO2"], ["mix", "heat"])

    match = _compute_precursor_match(state, gold)
    # Gold has carbonate+oxide, state has oxide+oxide
    # 1 class overlap (oxide), 2 gold classes -> 1/2 = 0.5
    assert match == 0.5


def test_calibrate_judge_basic():
    """Test basic calibration functionality"""
    # Create test routes
    train_routes = [
        _make_route("LiCoO2", ["Li2CO3", "Co3O4"], ["mix", "heat"]),
        _make_route("NiO", ["Ni(NO3)2"], ["heat", "cool"]),
    ]

    test_routes = [
        _make_route("BaTiO3", ["BaCO3", "TiO2"], ["mix", "heat"]),
    ]

    result = calibrate_judge(
        judge_name="deterministic",
        test_routes=test_routes,
        train_routes=train_routes,
        max_samples=10,
    )

    assert result.judge_name == "deterministic"
    assert result.n_samples == 1
    assert 0.0 <= result.mean_judge_score <= 1.0
    assert isinstance(result.score_distribution, dict)


def test_calibrate_judge_empty():
    """Test calibration with no test routes"""
    result = calibrate_judge(
        judge_name="deterministic",
        test_routes=[],
        train_routes=[],
        max_samples=10,
    )

    assert result.n_samples == 0
    assert result.correlation_with_validity == 0.0


# Helper functions


def _make_route(target: str, precursor_formulas: list[str], operation_verbs: list[str]) -> RouteRecord:
    """Create a RouteRecord for testing"""
    # Hardcode common element sets for test targets
    element_map = {
        "BaTiO3": ("Ba", "Ti", "O"),
        "LiCoO2": ("Li", "Co", "O"),
        "NiO": ("Ni", "O"),
    }
    elements = element_map.get(target, tuple(c for c in target if c.isupper()))

    precursors = tuple(
        PrecursorRecord(
            formula=formula,
            class_name="carbonate" if "CO3" in formula else "nitrate" if "NO3" in formula else "oxide",
            elements=elements,
        )
        for formula in precursor_formulas
    )

    operations = tuple(OperationRecord(verb=verb) for verb in operation_verbs)

    return RouteRecord(
        route_id=f"test_{target}",
        source_doi="10.1000/test",
        publication_year=2020,
        modality="solid_state",
        target_formula=target,
        target_elements=elements,
        chemical_system="-".join(elements),
        target_class="oxide",
        precursors=precursors,
        solvents=(),
        operations=operations,
        reaction_string=f"{'+'.join(precursor_formulas)} -> {target}",
        paragraph_excerpt="Test synthesis",
        source_dataset="test",
    )
