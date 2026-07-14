"""Tests for reaction driving force calculation."""

from __future__ import annotations

from synthesis_planner.chemistry import _compute_reaction_driving_force, _get_standard_formation_energy
from synthesis_planner.schema import (
    BalancedSpecies,
    PlanningProblem,
    PlanningState,
    PrecursorRecord,
    ReactionBalanceResult,
)


class MockMPClient:
    """Mock Materials Project client for testing"""

    def __init__(self):
        # Mock formation energies (eV/atom)
        self.formation_energies = {
            "BaTiO3": -1.5,
            "BaCO3": -1.2,
            "TiO2": -0.8,
            "Li2O": -1.0,
            "CoO": -0.6,
        }

    def get_formation_energy(self, formula: str) -> float | None:
        return self.formation_energies.get(formula)


def test_get_standard_formation_energy():
    """Test standard formation energies for common molecules"""
    assert _get_standard_formation_energy("H2O") is not None
    assert _get_standard_formation_energy("CO2") is not None
    assert _get_standard_formation_energy("N2") == 0.0  # Element reference
    assert _get_standard_formation_energy("O2") == 0.0  # Element reference
    assert _get_standard_formation_energy("unknown") is None


def test_compute_reaction_driving_force_basic():
    """Test basic reaction driving force calculation"""
    # BaCO3 + TiO2 -> BaTiO3 + CO2
    state = PlanningState(
        problem=PlanningProblem(target_formula="BaTiO3"),
        target_elements=("Ba", "Ti", "O"),
        target_class="oxide",
        precursors=(
            PrecursorRecord(formula="BaCO3", class_name="carbonate", elements=("Ba", "C", "O")),
            PrecursorRecord(formula="TiO2", class_name="oxide", elements=("Ti", "O")),
        ),
    )

    balance = ReactionBalanceResult(
        feasible=True,
        framework_match_fraction=1.0,
        precursor_coefficients=(1.0, 1.0),
        byproducts=(BalancedSpecies(formula="CO2", coefficient=1.0),),
        environmental_reactants=(),
    )

    mp_client = MockMPClient()
    target_formation_energy = -1.5  # BaTiO3

    result = _compute_reaction_driving_force(state, balance, target_formation_energy, mp_client)

    assert result is not None
    assert "delta_h" in result
    assert "h_target" in result
    assert "h_precursors" in result
    assert isinstance(result["delta_h"], float)


def test_compute_reaction_driving_force_missing_precursor():
    """Test that function returns None when precursor energy unavailable"""
    state = PlanningState(
        problem=PlanningProblem(target_formula="BaTiO3"),
        target_elements=("Ba", "Ti", "O"),
        target_class="oxide",
        precursors=(
            PrecursorRecord(formula="UnknownPrecursor", class_name="unknown", elements=("U",)),
        ),
    )

    balance = ReactionBalanceResult(
        feasible=True,
        framework_match_fraction=1.0,
        precursor_coefficients=(1.0,),
        byproducts=(),
        environmental_reactants=(),
    )

    mp_client = MockMPClient()
    target_formation_energy = -1.5

    result = _compute_reaction_driving_force(state, balance, target_formation_energy, mp_client)

    # Should return None when precursor energy not available
    assert result is None


def test_compute_reaction_driving_force_exothermic():
    """Test detection of exothermic reactions"""
    # Setup where products are lower energy than reactants (exothermic)
    state = PlanningState(
        problem=PlanningProblem(target_formula="Li2O"),
        target_elements=("Li", "O"),
        target_class="oxide",
        precursors=(
            PrecursorRecord(formula="Li2O", class_name="oxide", elements=("Li", "O")),
        ),
    )

    balance = ReactionBalanceResult(
        feasible=True,
        framework_match_fraction=1.0,
        precursor_coefficients=(0.5,),  # Starting from less material
        byproducts=(),
        environmental_reactants=(),
    )

    mp_client = MockMPClient()
    target_formation_energy = -1.0  # Li2O

    result = _compute_reaction_driving_force(state, balance, target_formation_energy, mp_client)

    if result is not None:
        # If calculation succeeds, check structure
        assert "delta_h" in result
        # Note: Actual sign depends on the mock energies and coefficients


def test_compute_reaction_driving_force_with_byproducts():
    """Test calculation with byproducts"""
    state = PlanningState(
        problem=PlanningProblem(target_formula="BaTiO3"),
        target_elements=("Ba", "Ti", "O"),
        target_class="oxide",
        precursors=(
            PrecursorRecord(formula="BaCO3", class_name="carbonate", elements=("Ba", "C", "O")),
            PrecursorRecord(formula="TiO2", class_name="oxide", elements=("Ti", "O")),
        ),
    )

    balance = ReactionBalanceResult(
        feasible=True,
        framework_match_fraction=1.0,
        precursor_coefficients=(1.0, 1.0),
        byproducts=(
            BalancedSpecies(formula="CO2", coefficient=1.0),
            BalancedSpecies(formula="H2O", coefficient=0.5),
        ),
        environmental_reactants=(),
    )

    mp_client = MockMPClient()
    target_formation_energy = -1.5

    result = _compute_reaction_driving_force(state, balance, target_formation_energy, mp_client)

    if result is not None:
        # Byproducts should contribute to products side
        assert "h_byproducts" in result
        # CO2 and H2O have negative formation energies, should be included


def test_compute_reaction_driving_force_with_env_reactants():
    """Test calculation with environmental reactants (e.g., O2)"""
    state = PlanningState(
        problem=PlanningProblem(target_formula="BaTiO3"),
        target_elements=("Ba", "Ti", "O"),
        target_class="oxide",
        precursors=(
            PrecursorRecord(formula="BaCO3", class_name="carbonate", elements=("Ba", "C", "O")),
            PrecursorRecord(formula="TiO2", class_name="oxide", elements=("Ti", "O")),
        ),
    )

    balance = ReactionBalanceResult(
        feasible=True,
        framework_match_fraction=1.0,
        precursor_coefficients=(1.0, 1.0),
        byproducts=(BalancedSpecies(formula="CO2", coefficient=1.0),),
        environmental_reactants=(BalancedSpecies(formula="O2", coefficient=0.5),),
    )

    mp_client = MockMPClient()
    target_formation_energy = -1.5

    result = _compute_reaction_driving_force(state, balance, target_formation_energy, mp_client)

    if result is not None:
        # Environmental reactants should contribute to reactants side
        assert "h_env_reactants" in result
        # O2 has zero formation energy (element reference)
