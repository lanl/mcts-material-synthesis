"""Tests for the thermodynamic provider seam (real-energy path for physics).

Covers the offline CachedThermoProvider, the eV/atom -> kJ/mol conversion, the
live-client MPThermoProvider adapter, the build_thermo_cache populate path, and
the end-to-end fact that a real-energy provider overrides the offline table
inside reaction_enthalpy (i.e. it changes route ranking).
"""

import json

from synthesis_planner.core.physics import (
    formation_enthalpy,
    reaction_enthalpy,
    target_stability,
)
from synthesis_planner.core.schema import (
    BalancedSpecies,
    PlanningProblem,
    PlanningState,
    PrecursorRecord,
    ReactionBalanceResult,
)
from synthesis_planner.data.materials_project import (
    EV_TO_KJ_PER_MOL,
    CachedThermoProvider,
    MPThermoProvider,
    build_thermo_cache,
    ev_per_atom_to_kj_per_mol,
)


def test_ev_per_atom_to_kj_per_mol_uses_atom_count():
    # BaTiO3 has 5 atoms/f.u.; -2.0 eV/atom -> -2 * 5 * 96.485 kJ/mol f.u.
    got = ev_per_atom_to_kj_per_mol(-2.0, "BaTiO3")
    assert abs(got - (-2.0 * 5 * EV_TO_KJ_PER_MOL)) < 1e-6


def test_ev_per_atom_to_kj_per_mol_none_for_unparseable():
    assert ev_per_atom_to_kj_per_mol(-1.0, "!!!") is None


def test_cached_provider_lookup_and_conversion():
    provider = CachedThermoProvider(
        {"BaTiO3": {"formation_energy_ev_per_atom": -3.1, "energy_above_hull": 0.0, "is_stable": True}}
    )
    assert "BaTiO3" in provider and len(provider) == 1
    # formation_enthalpy is returned in kJ/mol per formula unit.
    assert abs(provider.formation_enthalpy("BaTiO3") - (-3.1 * 5 * EV_TO_KJ_PER_MOL)) < 1e-6
    assert provider.energy_above_hull("BaTiO3") == 0.0
    assert provider.is_stable("BaTiO3") is True
    # Unknown -> None everywhere (never fabricated).
    assert provider.formation_enthalpy("SrTiO3") is None
    assert provider.energy_above_hull("SrTiO3") is None
    assert provider.is_stable("SrTiO3") is None


def test_cached_provider_json_round_trip(tmp_path):
    entries = {"CaO": {"formation_energy_ev_per_atom": -3.29, "energy_above_hull": 0.0, "is_stable": True}}
    path = tmp_path / "thermo.json"
    CachedThermoProvider(entries).to_json(path)
    reloaded = CachedThermoProvider.from_json(path)
    assert reloaded.energy_above_hull("CaO") == 0.0
    assert abs(reloaded.formation_enthalpy("CaO") - (-3.29 * 2 * EV_TO_KJ_PER_MOL)) < 1e-6


def test_provider_overrides_offline_table_in_reaction_enthalpy():
    """A real-energy provider must change the reaction enthalpy vs the offline
    table -- proving the seam actually reaches route ranking."""
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
    offline_dh, _ = reaction_enthalpy(state, balance)

    # Provider overrides only the target's formation enthalpy with a real value;
    # everything else falls through to the offline table.
    provider = CachedThermoProvider(
        {"BaTiO3": {"formation_energy_ev_per_atom": -3.5}}  # kJ/mol f.u. = -3.5*5*96.485
    )
    provider_dh, computable = reaction_enthalpy(state, balance, provider=provider)
    assert computable is True

    real_target_h = -3.5 * 5 * EV_TO_KJ_PER_MOL
    # Offline used the oxide-sum estimate for BaTiO3 (-548 - 944); provider swaps
    # in the real value, shifting dH by exactly the difference.
    expected_shift = real_target_h - (-548.0 - 944.0)
    assert abs((provider_dh - offline_dh) - expected_shift) < 1e-6

    # And target_stability now has hull info it lacked offline.
    assert target_stability("BaTiO3", provider=None) == (None, False)
    provider2 = CachedThermoProvider({"BaTiO3": {"energy_above_hull": 0.02}})
    assert target_stability("BaTiO3", provider=provider2) == (0.02, True)


def test_formation_enthalpy_prefers_provider_over_table():
    # CO2 is in the offline table (-393.5); a provider value must win.
    provider = CachedThermoProvider({"CO2": {"formation_energy_ev_per_atom": -1.0}})
    value, estimated = formation_enthalpy("CO2", provider=provider)
    assert estimated is False
    assert abs(value - (-1.0 * 3 * EV_TO_KJ_PER_MOL)) < 1e-6


class _FakeMPClient:
    """Minimal stand-in for MaterialsProjectClient (no network)."""

    def __init__(self, table):
        self._table = table  # formula -> (formation_ev_per_atom, hull_ev, is_stable)

    def get_formation_energy(self, formula):
        row = self._table.get(formula)
        return row[0] if row else None

    def get_hull_energy(self, formula):
        row = self._table.get(formula)
        return row[1] if row else None

    def get_thermodynamic_data(self, formula):
        from synthesis_planner.data.materials_project import ThermodynamicData

        row = self._table.get(formula)
        if not row:
            return ThermodynamicData(formula=formula)
        return ThermodynamicData(
            formula=formula,
            formation_energy_ev_per_atom=row[0],
            hull_energy_ev_per_atom=row[1],
            is_stable=row[2],
        )


def test_mp_thermo_provider_adapter_converts_units():
    client = _FakeMPClient({"BaTiO3": (-3.0, 0.0, True)})
    provider = MPThermoProvider(client)
    assert abs(provider.formation_enthalpy("BaTiO3") - (-3.0 * 5 * EV_TO_KJ_PER_MOL)) < 1e-6
    assert provider.energy_above_hull("BaTiO3") == 0.0
    assert provider.formation_enthalpy("SrTiO3") is None


def test_build_thermo_cache_populates_and_reloads(tmp_path):
    client = _FakeMPClient({"BaTiO3": (-3.0, 0.0, True), "SrTiO3": (-3.2, 0.01, True)})
    path = tmp_path / "cache.json"
    provider = build_thermo_cache(client, ["BaTiO3", "SrTiO3", "BaTiO3"], path)
    # Deduped, both present, written to disk.
    assert len(provider) == 2
    on_disk = json.loads(path.read_text())
    assert on_disk["BaTiO3"]["formation_energy_ev_per_atom"] == -3.0
    # Reload offline and confirm conversion round-trips.
    reloaded = CachedThermoProvider.from_json(path)
    assert abs(reloaded.formation_enthalpy("SrTiO3") - (-3.2 * 5 * EV_TO_KJ_PER_MOL)) < 1e-6
    assert reloaded.energy_above_hull("SrTiO3") == 0.01
