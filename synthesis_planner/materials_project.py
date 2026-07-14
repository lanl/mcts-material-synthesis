"""Materials Project API integration for thermodynamic data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .formula import parse_formula


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
