"""Materials Project API integration for thermodynamic data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..core.formula import parse_formula

# eV -> kJ/mol (per mole of the same entity). Materials Project reports formation
# energy in eV/atom; core/physics.py works in kJ/mol per formula unit. The
# conversion is: dHf[kJ/mol f.u.] = E_f[eV/atom] * n_atoms(f.u.) * EV_TO_KJ_PER_MOL.
EV_TO_KJ_PER_MOL = 96.48533212


@dataclass(frozen=True)
class ThermodynamicData:
    """Thermodynamic data from Materials Project"""
    formula: str
    hull_energy_ev_per_atom: float | None = None
    formation_energy_ev_per_atom: float | None = None
    decomposition_energy_ev_per_atom: float | None = None
    competing_phases: tuple[str, ...] = ()
    is_stable: bool = False


class MaterialsProjectClient:
    """Client for Materials Project API"""

    def __init__(self, api_key: str):
        """
        Initialize Materials Project client.

        Args:
            api_key: Materials Project API key (get from https://next-gen.materialsproject.org/api)
        """
        try:
            from mp_api.client import MPRester
            self.mp_rester = MPRester(api_key)
            self.enabled = True
        except ImportError:
            print("Warning: mp-api not installed. Materials Project integration disabled.")
            self.mp_rester = None
            self.enabled = False
        except Exception as e:
            print(f"Warning: Materials Project client initialization failed: {e}")
            self.mp_rester = None
            self.enabled = False

    def get_thermodynamic_data(self, formula: str) -> ThermodynamicData | None:
        """
        Get thermodynamic data for a formula.

        Args:
            formula: Chemical formula (e.g., "BaTiO3")

        Returns:
            ThermodynamicData or None if not available
        """
        if not self.enabled or not self.mp_rester:
            return None

        try:
            # Search for material by formula
            docs = self.mp_rester.thermo.search(
                formula=formula,
                fields=["formula_pretty", "energy_above_hull", "formation_energy_per_atom", "decomposition_energy", "is_stable"]
            )

            if not docs:
                return ThermodynamicData(formula=formula)

            # Take the most stable entry (lowest energy above hull)
            doc = min(docs, key=lambda d: d.energy_above_hull if d.energy_above_hull is not None else float('inf'))

            hull_energy = doc.energy_above_hull if hasattr(doc, 'energy_above_hull') else None
            formation_energy = doc.formation_energy_per_atom if hasattr(doc, 'formation_energy_per_atom') else None
            decomposition_energy = doc.decomposition_energy if hasattr(doc, 'decomposition_energy') else None
            is_stable = doc.is_stable if hasattr(doc, 'is_stable') else False

            # Get competing phases
            competing = self._get_competing_phases(formula)

            return ThermodynamicData(
                formula=formula,
                hull_energy_ev_per_atom=hull_energy,
                formation_energy_ev_per_atom=formation_energy,
                decomposition_energy_ev_per_atom=decomposition_energy,
                competing_phases=competing,
                is_stable=is_stable
            )

        except Exception as e:
            print(f"Warning: Failed to fetch thermodynamic data for {formula}: {e}")
            return ThermodynamicData(formula=formula)

    def _get_competing_phases(self, formula: str, max_phases: int = 5) -> tuple[str, ...]:
        """
        Get competing phases in the chemical system.

        Args:
            formula: Target formula
            max_phases: Maximum number of competing phases to return

        Returns:
            Tuple of competing phase formulas
        """
        if not self.enabled or not self.mp_rester:
            return ()

        try:
            # Parse formula to get elements
            parsed = parse_formula(formula)
            if not parsed:
                return ()

            elements = sorted(parsed.keys())
            if not elements:
                return ()

            # Get phase diagram for the chemical system
            # Note: phase diagram API may vary by mp-api version
            chemsys = "-".join(elements)

            # Search for stable phases in this chemical system
            docs = self.mp_rester.thermo.search(
                chemsys=chemsys,
                is_stable=True,
                fields=["formula_pretty", "energy_above_hull"]
            )

            if not docs:
                return ()

            # Exclude the target formula itself and return top competing phases
            competing = [
                doc.formula_pretty
                for doc in docs
                if doc.formula_pretty != formula and doc.formula_pretty
            ]

            return tuple(competing[:max_phases])

        except Exception as e:
            print(f"Warning: Failed to get competing phases for {formula}: {e}")
            return ()

    def get_formation_energy(self, formula: str) -> float | None:
        """
        Get formation energy for a formula.

        Args:
            formula: Chemical formula

        Returns:
            Formation energy in eV/atom or None
        """
        data = self.get_thermodynamic_data(formula)
        return data.formation_energy_ev_per_atom if data else None

    def get_hull_energy(self, formula: str) -> float | None:
        """
        Get energy above convex hull for a formula.

        Args:
            formula: Chemical formula

        Returns:
            Energy above hull in eV/atom or None (0 = stable)
        """
        data = self.get_thermodynamic_data(formula)
        return data.hull_energy_ev_per_atom if data else None

    def is_stable(self, formula: str) -> bool:
        """
        Check if a formula is thermodynamically stable.

        Args:
            formula: Chemical formula

        Returns:
            True if stable (on convex hull)
        """
        data = self.get_thermodynamic_data(formula)
        return data.is_stable if data else False


def create_mp_client_from_config(config: dict) -> MaterialsProjectClient | None:
    """
    Create Materials Project client from config dictionary.

    Args:
        config: Configuration dict with 'materials_project' section

    Returns:
        MaterialsProjectClient or None if disabled or no API key
    """
    mp_config = config.get("materials_project", {})

    if not mp_config.get("enable", False):
        return None

    api_key = mp_config.get("api_key", "")
    if not api_key:
        print("Warning: Materials Project API key not found in config")
        return None

    return MaterialsProjectClient(api_key)


# ---------------------------------------------------------------------------
# Thermodynamic provider seam for core/physics.py
#
# core/scoring.py passes ``config.physics_provider`` into the physics module,
# which duck-types two methods:
#   * ``formation_enthalpy(formula) -> float | None``  (kJ/mol per formula unit)
#         -- flows into the balanced *reaction* enthalpy, so it changes route
#            ranking (byproducts/precursors differ per route).
#   * ``energy_above_hull(formula) -> float | None``   (eV/atom, optional)
#         -- a *target-level* stability screen. NB: for a fixed target this is
#            constant across all routes, so it discriminates targets
#            (prospective screening), not routes within one target.
#
# Two concrete providers are supplied: a fully offline ``CachedThermoProvider``
# (reads a JSON energy dump) and a live ``MPThermoProvider`` (wraps
# MaterialsProjectClient). ``build_thermo_cache`` is the one-shot populate path:
# run it once on a networked machine with an MP key, then everything runs offline
# off the resulting JSON.
# ---------------------------------------------------------------------------


def _n_atoms(formula: str) -> Optional[int]:
    """Atoms per formula unit, or None if unparseable (no fabricated value)."""
    try:
        counts = parse_formula(formula)
    except Exception:
        return None
    if not counts:
        return None
    total = sum(counts.values())
    return total if total > 0 else None


def ev_per_atom_to_kj_per_mol(formation_ev_per_atom: float, formula: str) -> Optional[float]:
    """Convert an MP-style formation energy (eV/atom) to kJ/mol per formula unit.

    Returns None when the formula cannot be parsed for its atom count, so a bad
    formula never yields a fabricated enthalpy.
    """
    n = _n_atoms(formula)
    if n is None:
        return None
    return formation_ev_per_atom * n * EV_TO_KJ_PER_MOL


class CachedThermoProvider:
    """Offline thermodynamic provider backed by a JSON energy cache.

    The cache maps ``formula -> {formation_energy_ev_per_atom, energy_above_hull,
    is_stable}`` (any field optional). This is the drop-in that makes real MP/DFT
    energies available in an air-gapped environment: populate the cache once with
    ``build_thermo_cache`` (or any external dump), then construct from the file.

    Exposes the physics-seam contract: ``formation_enthalpy`` (kJ/mol f.u.) and
    ``energy_above_hull`` (eV/atom). Unknown formulas return None so the physics
    module stays neutral rather than guessing.
    """

    def __init__(self, entries: dict[str, dict[str, Any]] | None = None):
        self._entries: dict[str, dict[str, Any]] = dict(entries or {})

    @classmethod
    def from_json(cls, path: str | Path) -> "CachedThermoProvider":
        data = json.loads(Path(path).read_text())
        if not isinstance(data, dict):
            raise ValueError(f"thermo cache at {path} must be a JSON object")
        return cls(data)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self._entries, indent=2, sort_keys=True))

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, formula: str) -> bool:
        return formula in self._entries

    def formation_enthalpy(self, formula: str) -> Optional[float]:
        entry = self._entries.get(formula)
        if not entry:
            return None
        ev = entry.get("formation_energy_ev_per_atom")
        if ev is None:
            return None
        return ev_per_atom_to_kj_per_mol(float(ev), formula)

    def energy_above_hull(self, formula: str) -> Optional[float]:
        entry = self._entries.get(formula)
        if not entry:
            return None
        value = entry.get("energy_above_hull")
        return None if value is None else float(value)

    def is_stable(self, formula: str) -> Optional[bool]:
        entry = self._entries.get(formula)
        if not entry or "is_stable" not in entry:
            return None
        return bool(entry["is_stable"])


class MPThermoProvider:
    """Live physics-seam provider wrapping a MaterialsProjectClient.

    Thin adapter converting the client's eV/atom formation energy into the
    kJ/mol-per-formula-unit the physics module expects, and forwarding
    energy_above_hull unchanged. Every lookup hits the network, so for benchmark
    runs prefer building a cache first (``build_thermo_cache``).
    """

    def __init__(self, client: "MaterialsProjectClient"):
        self._client = client

    def formation_enthalpy(self, formula: str) -> Optional[float]:
        ev = self._client.get_formation_energy(formula)
        if ev is None:
            return None
        return ev_per_atom_to_kj_per_mol(float(ev), formula)

    def energy_above_hull(self, formula: str) -> Optional[float]:
        return self._client.get_hull_energy(formula)


def build_thermo_cache(
    client: "MaterialsProjectClient",
    formulas,
    path: str | Path,
) -> CachedThermoProvider:
    """One-shot populate: fetch MP thermo for ``formulas`` and write a JSON cache.

    Run once on a networked machine with an MP key; the resulting file feeds
    ``CachedThermoProvider.from_json`` offline. Formulas MP has no data for are
    stored with null fields so they are not re-fetched and stay neutral.
    """
    entries: dict[str, dict[str, Any]] = {}
    for formula in dict.fromkeys(formulas):  # dedupe, preserve order
        data = client.get_thermodynamic_data(formula)
        entries[formula] = {
            "formation_energy_ev_per_atom": getattr(data, "formation_energy_ev_per_atom", None) if data else None,
            "energy_above_hull": getattr(data, "hull_energy_ev_per_atom", None) if data else None,
            "is_stable": getattr(data, "is_stable", None) if data else None,
        }
    provider = CachedThermoProvider(entries)
    provider.to_json(path)
    return provider
