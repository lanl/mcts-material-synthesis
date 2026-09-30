"""Tests for the offline thermodynamic physics signal."""

from synthesis_planner.core.physics import (
    formation_enthalpy,
    reaction_enthalpy,
    stability_favorability,
    target_stability,
    thermo_favorability,
)
from synthesis_planner.core.schema import (
    BalancedSpecies,
    PlanningProblem,
    PlanningState,
    PrecursorRecord,
    ReactionBalanceResult,
)


def test_formation_enthalpy_table_lookup():
    value, estimated = formation_enthalpy("CO2")
    assert value == -393.5 and estimated is False


def test_formation_enthalpy_oxide_sum_estimate_covers_novel():
    # BaTiO3 is not in the table -> estimated from BaO + TiO2 contributions.
    value, estimated = formation_enthalpy("BaTiO3")
    assert estimated is True
    assert value == -548.0 + -944.0  # Ba + Ti cation-oxide contributions


def test_formation_enthalpy_none_for_unknown_nonoxide():
    value, estimated = formation_enthalpy("XeKr")
    assert value is None and estimated is False


def test_provider_seam_overrides_table():
    class Provider:
        def formation_enthalpy(self, formula):
            return -1234.0 if formula == "CO2" else None

    value, estimated = formation_enthalpy("CO2", provider=Provider())
    assert value == -1234.0 and estimated is False


def test_thermo_favorability_monotonic_and_neutral():
    assert thermo_favorability(None, computable=False) == 0.5
    downhill = thermo_favorability(-300.0, True)
    flat = thermo_favorability(0.0, True)
    uphill = thermo_favorability(+300.0, True)
    assert downhill > flat > uphill
    assert abs(flat - 0.5) < 1e-9


def test_reaction_enthalpy_balances_carbonate_route():
    # BaCO3 + TiO2 -> BaTiO3 + CO2 ; all species have (table or estimated) dHf.
    state = PlanningState(
        problem=PlanningProblem(target_formula="BaTiO3"),
        target_elements=("Ba", "Ti", "O"),
        target_class="oxide",
        stage="terminal",
        precursors=(
            PrecursorRecord("BaCO3", "carbonate", ("Ba", "C", "O")),
            PrecursorRecord("TiO2", "oxide", ("Ti", "O")),
        ),
    )
    balance = ReactionBalanceResult(
        feasible=True,
        framework_match_fraction=1.0,
        precursor_coefficients=(1.0, 1.0),
        byproducts=(BalancedSpecies(formula="CO2", coefficient=1.0),),
    )
    delta_h, computable = reaction_enthalpy(state, balance)
    assert computable is True
    # dH = [dHf(BaTiO3) + dHf(CO2)] - [dHf(BaCO3) + dHf(TiO2)]
    expected = ((-548.0 - 944.0) + -393.5) - (-1216.0 + -944.0)
    assert abs(delta_h - expected) < 1e-6


def test_reaction_enthalpy_not_computable_without_balance():
    state = PlanningState(
        problem=PlanningProblem(target_formula="BaTiO3"),
        target_elements=("Ba", "Ti", "O"),
        target_class="oxide",
    )
    delta_h, computable = reaction_enthalpy(state, None)
    assert delta_h is None and computable is False


# --- target-stability screen (real-energy provider only) ---------------------


def test_target_stability_none_without_provider_or_method():
    # No provider -> unavailable (offline table has no hull info).
    assert target_stability("BaTiO3", provider=None) == (None, False)

    class NoHull:
        def formation_enthalpy(self, formula):
            return -100.0

    # A provider lacking energy_above_hull stays unavailable.
    assert target_stability("BaTiO3", provider=NoHull()) == (None, False)


def test_target_stability_reads_provider_hull():
    class Provider:
        def energy_above_hull(self, formula):
            return 0.05 if formula == "BaTiO3" else None

    value, available = target_stability("BaTiO3", provider=Provider())
    assert available is True and value == 0.05
    assert target_stability("XYZ", provider=Provider()) == (None, False)


def test_stability_favorability_monotonic_and_neutral():
    assert stability_favorability(None, available=False) == 0.5
    on_hull = stability_favorability(0.0, True)
    metastable = stability_favorability(0.1, True)
    unstable = stability_favorability(0.5, True)
    assert on_hull == 1.0
    assert on_hull > metastable > unstable
    assert 0.0 < unstable < metastable
