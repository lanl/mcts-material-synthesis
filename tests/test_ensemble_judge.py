"""Tests for ensemble judge with uncertainty quantification."""

from synthesis_planner.judge import EnsembleJudge, build_judge
from synthesis_planner.schema import (
    HardCheckResult,
    OperationRecord,
    PlanningProblem,
    PlanningState,
    PrecursorRecord,
    ReactionBalanceResult,
    RouteRecord,
)


def test_ensemble_judge_creation():
    """Test that ensemble judge can be created"""
    judge = build_judge("ensemble", {"num_variants": 3})
    assert isinstance(judge, EnsembleJudge)
    assert judge.name == "ensemble"
    assert len(judge.variant_judges) == 3


def test_ensemble_judge_with_deterministic():
    """Test ensemble judge with deterministic base judge"""
    judge = EnsembleJudge({"base_judge": "deterministic", "num_variants": 3})

    state = _make_state()
    analogs = _make_analogs()
    hard_checks = _make_hard_checks()

    result = judge.evaluate(state, analogs, hard_checks)

    # Should return a valid result
    assert 0.0 <= result.score <= 1.0
    assert isinstance(result.notes, tuple)
    assert isinstance(result.flags, tuple)
    # With deterministic judges, uncertainty should be low (identical results)
    assert result.uncertainty >= 0.0


def test_ensemble_judge_uncertainty_computation():
    """Test that ensemble judge computes uncertainty from disagreement"""
    # With deterministic judges, all should agree (low uncertainty)
    judge = EnsembleJudge({"base_judge": "deterministic", "num_variants": 3})

    state = _make_state()
    analogs = _make_analogs()
    hard_checks = _make_hard_checks()

    result = judge.evaluate(state, analogs, hard_checks)

    # Uncertainty should exist (even if low for deterministic)
    assert 0.0 <= result.uncertainty <= 1.0


def test_ensemble_judge_prompt_variants():
    """Test that prompt variants are defined"""
    assert "conservative" in EnsembleJudge.PROMPT_VARIANTS
    assert "optimistic" in EnsembleJudge.PROMPT_VARIANTS
    assert "skeptical" in EnsembleJudge.PROMPT_VARIANTS

    # Check that variants have different prompts
    conservative = EnsembleJudge.PROMPT_VARIANTS["conservative"]
    optimistic = EnsembleJudge.PROMPT_VARIANTS["optimistic"]
    skeptical = EnsembleJudge.PROMPT_VARIANTS["skeptical"]

    assert conservative != optimistic
    assert optimistic != skeptical
    assert "skeptical" in skeptical.lower()
    assert "conservative" in conservative.lower() or "skeptical" in conservative.lower()


def test_ensemble_judge_merges_notes_and_flags():
    """Test that ensemble judge merges notes and flags from variants"""
    judge = EnsembleJudge({"base_judge": "deterministic", "num_variants": 3})

    state = _make_state()
    analogs = _make_analogs()
    hard_checks = _make_hard_checks()

    result = judge.evaluate(state, analogs, hard_checks)

    # Should have merged notes
    assert len(result.notes) > 0
    # Should deduplicate notes
    assert len(result.notes) == len(set(result.notes))


def test_ensemble_judge_handles_variant_failure():
    """Test that ensemble judge continues when a variant fails"""
    # Create ensemble with config that might cause some variants to fail gracefully
    judge = EnsembleJudge({"base_judge": "deterministic", "num_variants": 2})

    state = _make_state()
    analogs = []  # Empty analogs - might cause issues but should be handled
    hard_checks = _make_hard_checks()

    # Should not raise exception
    result = judge.evaluate(state, analogs, hard_checks)
    assert result is not None
    assert 0.0 <= result.score <= 1.0


# Helper functions


def _make_state() -> PlanningState:
    """Create a test planning state"""
    return PlanningState(
        problem=PlanningProblem(target_formula="BaTiO3", modality="solid_state"),
        target_elements=("Ba", "Ti", "O"),
        target_class="oxide",
        stage="terminal",
        precursors=(
            PrecursorRecord(formula="BaCO3", class_name="carbonate", elements=("Ba", "C", "O")),
            PrecursorRecord(formula="TiO2", class_name="oxide", elements=("Ti", "O")),
        ),
        solvents=(),
        operations=(
            OperationRecord(verb="mix"),
            OperationRecord(verb="grind"),
            OperationRecord(verb="heat"),
        ),
    )


def _make_analogs() -> list[tuple[float, RouteRecord]]:
    """Create test analogs"""
    route = RouteRecord(
        route_id="test_analog",
        source_doi="10.1000/test",
        publication_year=2020,
        modality="solid_state",
        target_formula="SrTiO3",
        target_elements=("Sr", "Ti", "O"),
        chemical_system="Sr-Ti-O",
        target_class="oxide",
        precursors=(
            PrecursorRecord(formula="SrCO3", class_name="carbonate", elements=("Sr", "C", "O")),
            PrecursorRecord(formula="TiO2", class_name="oxide", elements=("Ti", "O")),
        ),
        solvents=(),
        operations=(OperationRecord(verb="mix"), OperationRecord(verb="heat")),
        reaction_string="SrCO3 + TiO2 -> SrTiO3 + CO2",
        paragraph_excerpt="Test analog synthesis",
        source_dataset="test",
    )
    return [(8.5, route)]


def _make_hard_checks() -> HardCheckResult:
    """Create test hard check result"""
    return HardCheckResult(
        valid=True,
        flags=(),
        notes=("Route is valid.",),
        coverage_fraction=1.0,
        blocking_flags=(),
        reaction_balance=ReactionBalanceResult(
            feasible=True,
            framework_match_fraction=1.0,
            precursor_coefficients=(1.0, 1.0),
        ),
    )
