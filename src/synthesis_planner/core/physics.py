"""Offline thermodynamic physics signal for synthesis routes.

Provides a *non-analog* reward signal: the thermodynamic driving force of the
balanced synthesis reaction, from standard formation enthalpies. Because it is
computed from the reaction (not from retrieved literature), it is available even
for novel targets with no close analog.

Three-tier formation-enthalpy source, in priority order:
  1. an injected ``provider`` (the Materials Project / DFT / ML seam) -- if it
     returns a value for a formula, that wins;
  2. a curated table of standard formation enthalpies (kJ/mol, ~298 K) for common
     byproducts, precursors and binary oxides (approximate literature values);
  3. an **oxide-sum estimator** -- for any oxide of tabulated cations (including
     compositions absent from the corpus), estimate the formation enthalpy as the
     stoichiometric sum of its constituent binary oxides. This is what gives novel
     coverage.

Honest scope: with offline data the oxide-sum estimate makes the *reaction*
enthalpy (precursor decomposition / byproduct release) meaningful, but it cannot
distinguish a stable ternary from its binary mixture -- true novel-target
*stability* needs the provider seam (real MP/DFT energies). Documented so the
signal is not over-claimed.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from .formula import parse_formula

# Standard formation enthalpies dHf (kJ / mol formula unit, ~298 K). Approximate
# literature values; enough to rank reaction favorability, not for absolute
# thermochemistry.
_FORMATION_ENTHALPY: dict[str, float] = {
    # Gaseous / byproducts
    "O2": 0.0, "N2": 0.0, "H2": 0.0, "Cl2": 0.0,
    "CO2": -393.5, "CO": -110.5, "H2O": -241.8, "NH3": -45.9,
    "NO2": 33.2, "NO": 91.3, "SO2": -296.8, "SO3": -395.7,
    # Binary oxides
    "TiO2": -944.0, "BaO": -548.0, "SrO": -592.0, "CaO": -635.1, "MgO": -601.6,
    "Fe2O3": -824.2, "Fe3O4": -1118.4, "FeO": -272.0, "CoO": -237.9, "Co3O4": -891.0,
    "NiO": -239.7, "ZnO": -350.5, "CuO": -157.3, "Cu2O": -168.6, "MnO2": -520.0,
    "Mn2O3": -959.0, "Mn3O4": -1387.0, "Al2O3": -1675.7, "SiO2": -910.7, "ZrO2": -1080.0,
    "La2O3": -1793.7, "Y2O3": -1905.3, "Li2O": -597.9, "Na2O": -414.2, "K2O": -361.5,
    "Cr2O3": -1139.7, "V2O5": -1550.6, "Nb2O5": -1899.5, "Ta2O5": -2046.0,
    "WO3": -842.9, "MoO3": -745.1, "Bi2O3": -573.9, "PbO": -217.3, "SnO2": -577.6,
    "GeO2": -580.0, "SnO": -280.7,
    # Carbonates
    "BaCO3": -1216.0, "SrCO3": -1220.0, "CaCO3": -1207.6, "MgCO3": -1095.8,
    "Li2CO3": -1215.9, "Na2CO3": -1130.7, "K2CO3": -1150.0, "MnCO3": -894.1,
    "CoCO3": -722.6, "NiCO3": -689.0, "ZnCO3": -812.8, "FeCO3": -740.6,
    # Nitrates (anhydrous)
    "Ba(NO3)2": -992.1, "Sr(NO3)2": -978.2, "Ca(NO3)2": -938.4, "Mg(NO3)2": -790.7,
    "Co(NO3)2": -420.5, "Ni(NO3)2": -415.0, "Zn(NO3)2": -483.7, "Cu(NO3)2": -302.9,
    "Fe(NO3)3": -670.0, "Al(NO3)3": -1033.0, "AgNO3": -124.4, "LiNO3": -483.1,
    "NaNO3": -467.9, "KNO3": -494.6,
    # Hydroxides
    "NaOH": -425.6, "KOH": -424.6, "LiOH": -487.5, "Ca(OH)2": -985.2,
    "Mg(OH)2": -924.5, "Al(OH)3": -1277.0, "Fe(OH)3": -823.0,
}

# Per-cation contribution for the oxide-sum estimator: element -> (dHf of its
# reference binary oxide per mole of cation, oxygen atoms consumed per cation).
# e.g. Ti -> TiO2 -> (-944.0 per Ti, 2 O per Ti); La -> La2O3 -> (-896.85, 1.5).
_CATION_OXIDE: dict[str, tuple[float, float]] = {
    "Ti": (-944.0, 2.0), "Ba": (-548.0, 1.0), "Sr": (-592.0, 1.0), "Ca": (-635.1, 1.0),
    "Mg": (-601.6, 1.0), "Fe": (-412.1, 1.5), "Co": (-297.0, 1.333), "Ni": (-239.7, 1.0),
    "Zn": (-350.5, 1.0), "Cu": (-157.3, 1.0), "Mn": (-520.0, 2.0), "Al": (-837.85, 1.5),
    "Si": (-910.7, 2.0), "Zr": (-1080.0, 2.0), "La": (-896.85, 1.5), "Y": (-952.65, 1.5),
    "Li": (-298.95, 0.5), "Na": (-207.1, 0.5), "K": (-180.75, 0.5), "Cr": (-569.85, 1.5),
    "V": (-310.12, 2.5), "Nb": (-949.75, 2.5), "Ta": (-1023.0, 2.5), "W": (-842.9, 3.0),
    "Mo": (-745.1, 3.0), "Bi": (-286.95, 1.5), "Pb": (-217.3, 1.0), "Sn": (-577.6, 2.0),
}

_ANIONS = frozenset({"O", "H", "C", "N", "S", "P", "F", "Cl", "Br", "I"})

# kJ/mol scale for mapping reaction enthalpy to a bounded [0,1] favorability.
_FAVORABILITY_SCALE_KJ = 300.0

# eV/atom scale for mapping energy-above-hull to a bounded [0,1] stability score.
# ~0.1 eV/atom is a common "metastable but plausibly synthesizable" threshold.
_STABILITY_SCALE_EV = 0.1


def formation_enthalpy(formula: str, provider: Optional[Any] = None) -> tuple[Optional[float], bool]:
    """Return (dHf in kJ/mol, is_estimated) for a formula, or (None, False).

    Priority: injected provider (MP/DFT seam) -> curated table -> oxide-sum
    estimator (for novel oxides). ``is_estimated`` flags the oxide-sum fallback.
    """
    if provider is not None:
        try:
            value = provider.formation_enthalpy(formula)
        except Exception:
            value = None
        if value is not None:
            return float(value), False
    if formula in _FORMATION_ENTHALPY:
        return _FORMATION_ENTHALPY[formula], False
    estimate = _oxide_sum_estimate(formula)
    if estimate is not None:
        return estimate, True
    return None, False


def _oxide_sum_estimate(formula: str) -> Optional[float]:
    """Estimate an oxide's dHf as the sum of its constituent binary oxides.

    Covers any oxide built from tabulated cations, including novel compositions.
    Returns None for non-oxides or unknown cations (no fabricated value).
    """
    try:
        counts = parse_formula(formula)
    except Exception:
        return None
    if counts.get("O", 0) <= 0:
        return None
    cations = {el: n for el, n in counts.items() if el not in _ANIONS}
    if not cations or any(el not in _CATION_OXIDE for el in cations):
        return None
    total = 0.0
    for el, n in cations.items():
        dhf_per_cation, _o_per_cation = _CATION_OXIDE[el]
        total += n * dhf_per_cation
    return total


def reaction_enthalpy(state, balance, provider: Optional[Any] = None) -> tuple[Optional[float], bool]:
    """Balanced-reaction enthalpy dH (kJ per mole target), and whether computable.

    dH = [dHf(target) + sum dHf(byproducts)] - [sum dHf(precursors) + sum dHf(env)],
    on a 1-mole-target basis. Returns (None, False) if any required species lacks
    a formation enthalpy (so the caller can stay neutral rather than guess).
    """
    if balance is None or not getattr(balance, "feasible", False):
        return None, False
    coeffs = list(balance.precursor_coefficients or ())
    precursors = list(state.precursors)
    if len(coeffs) != len(precursors):
        return None, False

    any_estimated = False

    def _need(formula: str) -> Optional[float]:
        nonlocal any_estimated
        value, estimated = formation_enthalpy(formula, provider)
        any_estimated = any_estimated or estimated
        return value

    target_h = _need(state.problem.target_formula)
    if target_h is None:
        return None, False

    products = target_h
    for species in balance.byproducts:
        h = _need(species.formula)
        if h is None:
            return None, False
        products += species.coefficient * h

    reactants = 0.0
    for coeff, precursor in zip(coeffs, precursors):
        h = _need(precursor.formula)
        if h is None:
            return None, False
        reactants += coeff * h
    for species in getattr(balance, "environmental_reactants", ()) or ():
        h = _need(species.formula)
        if h is None:
            return None, False
        reactants += species.coefficient * h

    return products - reactants, True


def thermo_favorability(delta_h_kj: Optional[float], computable: bool) -> float:
    """Map reaction dH to a bounded [0,1] favorability (higher = more downhill).

    Neutral 0.5 when the enthalpy is not computable, so an unknown never
    penalizes a route. dH = -SCALE -> ~0.73, dH = 0 -> 0.5, dH = +SCALE -> ~0.27
    via a logistic in dH/SCALE; a genuine, monotonic energetic gradient.
    """
    if not computable or delta_h_kj is None:
        return 0.5
    return 1.0 / (1.0 + math.exp(delta_h_kj / _FAVORABILITY_SCALE_KJ))


def target_stability(formula: str, provider: Optional[Any] = None) -> tuple[Optional[float], bool]:
    """Return (energy_above_hull in eV/atom, available) for a target formula.

    Reads ``provider.energy_above_hull(formula)`` when the provider exposes it
    (the real MP/DFT stability signal). Returns (None, False) otherwise -- the
    offline table has no hull information, so this is *only* meaningful with a
    real-energy provider (a populated CachedThermoProvider or a live MP client).
    """
    if provider is None or not hasattr(provider, "energy_above_hull"):
        return None, False
    try:
        value = provider.energy_above_hull(formula)
    except Exception:
        value = None
    if value is None:
        return None, False
    return float(value), True


def stability_favorability(e_above_hull_ev: Optional[float], available: bool) -> float:
    """Map energy-above-hull (eV/atom) to a bounded [0,1] makeability score.

    On-hull (0 eV) -> 1.0; larger e_above_hull decays toward 0; neutral 0.5 when
    unavailable so an unknown never penalizes. NB: for a *fixed target* this is
    constant across all candidate routes, so it discriminates *targets*
    (prospective makeability screening / cross-target ranking), not routes within
    one target. It is deliberately not folded into the per-route retrospective
    reward, where it would be a route-invariant constant.
    """
    if not available or e_above_hull_ev is None:
        return 0.5
    return math.exp(-max(0.0, e_above_hull_ev) / _STABILITY_SCALE_EV)
