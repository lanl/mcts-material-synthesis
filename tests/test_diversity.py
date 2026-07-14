"""Tests for diversity-adjusted portfolio selection."""

from synthesis_planner.planner import _route_similarity, _select_portfolio
from synthesis_planner.schema import (
    HardCheckResult,
    JudgeResult,
    NumericRange,
    OperationRecord,
    PlannedRoute,
    PrecursorRecord,
    ScoreBreakdown,
    ThermoAnalysisResult,
)


def test_route_similarity_identical():
    """Identical routes should have similarity 1.0"""
    route1 = _make_route(["BaCO3", "TiO2"], ["mix", "grind", "heat"], 1000.0, score=0.8)
    route2 = _make_route(["BaCO3", "TiO2"], ["mix", "grind", "heat"], 1000.0, score=0.8)

    similarity = _route_similarity(route1, route2)
    assert similarity == 1.0


def test_route_similarity_different_precursors():
    """Different precursors should reduce similarity"""
    route1 = _make_route(["BaCO3", "TiO2"], ["mix", "grind", "heat"], 1000.0, score=0.8)
    route2 = _make_route(["BaO", "TiO2"], ["mix", "grind", "heat"], 1000.0, score=0.8)

    similarity = _route_similarity(route1, route2)
    # Precursors differ: BaCO3 vs BaO
    # 1 overlap (TiO2), 3 unique (BaCO3, BaO, TiO2) -> Jaccard = 1/3 = 0.33
    # Operations same -> 1.0
    # Temp same -> 1.0
    # Weighted: 0.4 * 0.33 + 0.4 * 1.0 + 0.2 * 1.0 = 0.132 + 0.4 + 0.2 = 0.732
    assert 0.7 < similarity < 0.8


def test_route_similarity_different_operations():
    """Different operations should reduce similarity"""
    route1 = _make_route(["BaCO3", "TiO2"], ["mix", "grind", "heat"], 1000.0, score=0.8)
    route2 = _make_route(["BaCO3", "TiO2"], ["mix", "wash", "dry"], 1000.0, score=0.8)

    similarity = _route_similarity(route1, route2)
    # Precursors same -> 1.0
    # Operations differ significantly
    # Temp same -> 1.0
    assert similarity < 0.8


def test_route_similarity_different_temperature():
    """Different temperatures should reduce similarity"""
    route1 = _make_route(["BaCO3", "TiO2"], ["mix", "grind", "heat"], 800.0, score=0.8)
    route2 = _make_route(["BaCO3", "TiO2"], ["mix", "grind", "heat"], 1200.0, score=0.8)

    similarity = _route_similarity(route1, route2)
    # Precursors same -> 1.0
    # Operations same -> 1.0
    # Temp diff 400°C -> temp_sim = 1 - 400/200 = -1 -> max(0, -1) = 0
    # Weighted: 0.4 + 0.4 + 0.2 * 0 = 0.8
    assert 0.7 < similarity < 0.9


def test_select_portfolio_diversity():
    """Portfolio should select diverse routes"""
    # Create routes with same score but different characteristics
    route1 = _make_route(["BaCO3", "TiO2"], ["mix", "grind", "heat"], 1000.0, score=0.9)
    route2 = _make_route(["BaCO3", "TiO2"], ["mix", "grind", "heat"], 1000.0, score=0.9)  # Duplicate
    route3 = _make_route(["BaO", "TiO2"], ["mix", "heat"], 1200.0, score=0.85)  # Different
    route4 = _make_route(["Ba(OH)2", "TiO2"], ["mix", "dry", "heat"], 900.0, score=0.8)  # Different

    portfolio = _select_portfolio([route1, route2, route3, route4], top_k=3, diversity_threshold=0.7)

    # Should filter duplicates and select diverse routes
    assert len(portfolio) == 3
    # route1 or route2 (both identical, one will be selected)
    assert route1 in portfolio or route2 in portfolio
    assert route3 in portfolio
    assert route4 in portfolio


def test_select_portfolio_high_score_override():
    """High-scoring near-duplicate should be included if significantly better"""
    route1 = _make_route(["BaCO3", "TiO2"], ["mix", "grind", "heat"], 1000.0, score=0.7)
    route2 = _make_route(["BaCO3", "TiO2"], ["mix", "grind", "heat"], 1000.0, score=0.85)  # Much better

    portfolio = _select_portfolio([route1, route2], top_k=2, diversity_threshold=0.7)

    # route2 is duplicate but 21% better score
    # Since they're sorted by score, route2 comes first and is selected
    # route1 is then filtered as duplicate
    assert len(portfolio) >= 1
    assert route2 in portfolio  # Higher score selected


def test_select_portfolio_empty():
    """Empty input should return empty portfolio"""
    portfolio = _select_portfolio([], top_k=5)
    assert portfolio == []


def test_select_portfolio_fewer_than_k():
    """Should return all routes if fewer than k diverse routes"""
    route1 = _make_route(["BaCO3", "TiO2"], ["mix", "grind", "heat"], 1000.0, score=0.9)
    route2 = _make_route(["BaO", "TiO2"], ["mix", "heat"], 1200.0, score=0.85)

    portfolio = _select_portfolio([route1, route2], top_k=10)
    assert len(portfolio) == 2


# Helper function

def _make_route(
    precursor_formulas: list[str],
    operation_verbs: list[str],
    temperature_c: float,
    score: float
) -> PlannedRoute:
    """Create a PlannedRoute for testing"""
    precursors = tuple(
        PrecursorRecord(formula=f, class_name="test", elements=())
        for f in precursor_formulas
    )

    operations = []
    for verb in operation_verbs:
        if verb == "heat":
            operations.append(
                OperationRecord(
                    verb=verb,
                    temperature_c=NumericRange(minimum=temperature_c, maximum=temperature_c)
                )
            )
        else:
            operations.append(OperationRecord(verb=verb))

    return PlannedRoute(
        target_formula="BaTiO3",
        modality="solid_state",
        precursors=precursors,
        solvents=(),
        operations=tuple(operations),
        evidence_dois=(),
        analog_targets=(),
        hard_checks=HardCheckResult(
            valid=True, flags=(), notes=(), coverage_fraction=1.0, blocking_flags=()
        ),
        score=ScoreBreakdown(
            validity=score, stoich=score, precursor=score, thermo=score,
            retrieval=score, condition=score, llm=score,
            cost=0.0, hazard=0.0, complexity=0.0,
            total=score
        ),
        thermo=ThermoAnalysisResult(
            score=score, gas_release_moles=0.0, gas_uptake_moles=0.0,
            byproduct_count=0, decomposition_match=1.0, redox_match=1.0, notes=()
        ),
        judge=JudgeResult(score=score, notes=(), flags=()),
        mcts_value=score
    )
