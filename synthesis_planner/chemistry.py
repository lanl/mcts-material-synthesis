"""Route-level balancing, redox checks, and thermodynamic features."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations

from .formula import parse_formula, safe_required_target_elements
from .schema import BalancedSpecies, ReactionBalanceResult, RedoxAnalysisResult, ThermoAnalysisResult, PlanningState

VOLATILE_ELEMENTS = {"H", "C", "N", "O", "F", "Cl", "Br", "I"}
GAS_FORMULAS = {"CO2", "H2O", "N2", "NO2", "NO", "NH3", "HCl", "HBr", "HI", "HF", "SO2", "O2", "Cl2", "Br2", "I2", "H2"}
COMMON_BYPRODUCT_FORMULAS = ("CO2", "H2O", "NH3", "NO2", "NO", "N2", "HCl", "HBr", "HI", "HF", "SO2", "O2", "Cl2", "Br2", "I2", "H2")
OXIDIZING_ATMOSPHERES = {"air", "oxygen", "o2"}
REDUCING_ATMOSPHERES = {"h2", "hydrogen", "forming gas", "nh3", "ammonia", "co"}
INERT_ATMOSPHERES = {"n2", "nitrogen", "ar", "argon", "he", "helium"}
RECOGNIZED_ATMOSPHERES = OXIDIZING_ATMOSPHERES | REDUCING_ATMOSPHERES | INERT_ATMOSPHERES | {"vacuum", "inert"}
ALKALI_METALS = {"Li", "Na", "K", "Rb", "Cs"}
ALKALINE_EARTH_METALS = {"Be", "Mg", "Ca", "Sr", "Ba"}
FIXED_POSITIVE_STATES = {
    "Ag": 1,
    "Al": 3,
    "Cd": 2,
    "Ga": 3,
    "In": 3,
    "La": 3,
    "Sc": 3,
    "Y": 3,
    "Zn": 2,
}
DECOMPOSABLE_OXIDE_PRECURSORS = {"carbonate", "nitrate", "hydroxide", "oxalate", "acetate"}
SOLUTION_FRIENDLY_PRECURSORS = {"nitrate", "chloride", "bromide", "iodide", "acetate"}


@dataclass(frozen=True)
class _CandidateSpecies:
    formula: str
    role: str
    vector: tuple[Fraction, ...]


def balance_route(state: PlanningState) -> ReactionBalanceResult:
    target_counts = _fractional_counts(state.problem.target_formula)
    framework_elements = sorted(safe_required_target_elements(state.problem.target_formula))
    if not framework_elements:
        framework_elements = sorted(element for element in target_counts if element not in {"O", "H"})

    if not state.precursors or not framework_elements:
        return ReactionBalanceResult(
            feasible=False,
            framework_match_fraction=0.0,
            precursor_coefficients=tuple(0.0 for _ in state.precursors),
            residual_elements={},
        )

    precursor_matrix = [
        [_fractional_counts(precursor.formula).get(element, Fraction(0, 1)) for precursor in state.precursors]
        for element in framework_elements
    ]
    target_vector = [target_counts.get(element, Fraction(0, 1)) for element in framework_elements]
    precursor_coefficients = _solve_basic_feasible(precursor_matrix, target_vector)
    if precursor_coefficients is None:
        return ReactionBalanceResult(
            feasible=False,
            framework_match_fraction=_coverage_fraction(state, framework_elements),
            precursor_coefficients=tuple(0.0 for _ in state.precursors),
            residual_elements={},
        )

    precursor_totals = _combine_species(state.precursors, precursor_coefficients)
    residual = {
        element: precursor_totals.get(element, Fraction(0, 1)) - target_counts.get(element, Fraction(0, 1))
        for element in sorted(set(precursor_totals) | set(target_counts))
    }
    residual = {element: amount for element, amount in residual.items() if amount != 0}

    nonvolatile_residual = {element: amount for element, amount in residual.items() if element not in VOLATILE_ELEMENTS}
    if nonvolatile_residual:
        return ReactionBalanceResult(
            feasible=False,
            framework_match_fraction=1.0,
            precursor_coefficients=tuple(float(value) for value in precursor_coefficients),
            unused_precursors=_unused_precursors(state, precursor_coefficients),
            residual_elements={element: float(amount) for element, amount in nonvolatile_residual.items()},
        )

    products, reactants = _solve_volatile_residual(state, residual)
    if products is None or reactants is None:
        return ReactionBalanceResult(
            feasible=False,
            framework_match_fraction=1.0,
            precursor_coefficients=tuple(float(value) for value in precursor_coefficients),
            unused_precursors=_unused_precursors(state, precursor_coefficients),
            residual_elements={element: float(amount) for element, amount in residual.items()},
        )

    equation = _format_equation(state, precursor_coefficients, reactants, products)
    return ReactionBalanceResult(
        feasible=True,
        framework_match_fraction=1.0,
        precursor_coefficients=tuple(float(value) for value in precursor_coefficients),
        environmental_reactants=tuple(BalancedSpecies(formula=formula, coefficient=float(value)) for formula, value in reactants),
        byproducts=tuple(BalancedSpecies(formula=formula, coefficient=float(value)) for formula, value in products),
        unused_precursors=_unused_precursors(state, precursor_coefficients),
        residual_elements={},
        equation=equation,
    )


def analyze_redox(state: PlanningState, balance: ReactionBalanceResult) -> RedoxAnalysisResult:
    focus_elements = _redox_focus_elements(state.problem.target_formula, state.target_class)
    if not focus_elements or not balance.precursor_coefficients:
        return RedoxAnalysisResult(
            target_charge=None,
            precursor_charge=None,
            required_direction="unknown",
            environment_support="unknown",
            notes=("No redox-active target framework was available for analysis.",),
            flags=(),
        )

    target_charge = _infer_focus_charge_sum(state.problem.target_formula, focus_elements)
    precursor_charge = Fraction(0, 1)
    charge_observations = 0
    for precursor, coefficient in zip(state.precursors, balance.precursor_coefficients):
        if coefficient <= 0.0:
            continue
        focus_charge = _infer_focus_charge_sum(precursor.formula, focus_elements, precursor.class_name)
        if focus_charge is None:
            continue
        precursor_charge += _to_fraction(coefficient) * focus_charge
        charge_observations += 1

    if target_charge is None or charge_observations == 0:
        return RedoxAnalysisResult(
            target_charge=float(target_charge) if target_charge is not None else None,
            precursor_charge=float(precursor_charge) if charge_observations else None,
            required_direction="unknown",
            environment_support="unknown",
            notes=("Oxidation-state inference was ambiguous for the selected precursor set.",),
            flags=(),
        )

    delta = target_charge - precursor_charge
    notes = []
    flags = []
    if delta == 0:
        direction = "none"
        support = "supported"
        notes.append("The precursor charge budget already matches the target framework charge.")
    elif delta > 0:
        direction = "oxidation"
        support = "supported" if _supports_oxidation(state, balance) else "unsupported"
        notes.append("The target requires net oxidation relative to the precursor charge state.")
        if support == "unsupported":
            flags.append("missing_oxidant")
    else:
        direction = "reduction"
        support = "supported" if _supports_reduction(state, balance) else "unsupported"
        notes.append("The target requires net reduction relative to the precursor charge state.")
        if support == "unsupported":
            flags.append("missing_reductant")

    if direction == "reduction" and _has_oxidizing_atmosphere(state):
        flags.append("oxidizing_atmosphere_mismatch")
        support = "unsupported"
        notes.append("An oxidizing atmosphere conflicts with the inferred reduction requirement.")
    if direction == "oxidation" and _has_reducing_atmosphere(state):
        flags.append("reducing_atmosphere_mismatch")
        support = "unsupported"
        notes.append("A reducing atmosphere conflicts with the inferred oxidation requirement.")

    return RedoxAnalysisResult(
        target_charge=float(target_charge),
        precursor_charge=float(precursor_charge),
        required_direction=direction,
        environment_support=support,
        notes=tuple(dict.fromkeys(notes)),
        flags=tuple(dict.fromkeys(flags)),
    )


def _compute_reaction_driving_force(
    state: PlanningState,
    balance: ReactionBalanceResult,
    target_formation_energy: float,
    mp_client
) -> dict | None:
    """
    Compute reaction driving force (ΔH_rxn) using formation energies.

    ΔH_rxn = (H_products + H_byproducts) - (H_precursors + H_env_reactants)

    Args:
        state: Planning state with precursors
        balance: Balanced reaction result with coefficients
        target_formation_energy: Formation energy of target (eV/atom)
        mp_client: Materials Project client

    Returns:
        Dict with delta_h (total reaction enthalpy in eV) or None if calculation fails
    """
    try:
        # Get formation energies for all precursors
        precursor_energies = []
        for precursor in state.precursors:
            energy = mp_client.get_formation_energy(precursor.formula)
            if energy is None:
                # Can't compute without all precursor energies
                return None
            precursor_energies.append(energy)

        # Get formation energies for byproducts (if any)
        byproduct_energies = []
        for byproduct in balance.byproducts:
            energy = mp_client.get_formation_energy(byproduct.formula)
            # Byproducts might not have formation energies (e.g., CO2, H2O)
            # Use standard values if not in MP
            if energy is None:
                energy = _get_standard_formation_energy(byproduct.formula)
            if energy is not None:
                byproduct_energies.append((energy, byproduct.coefficient))

        # Get formation energies for environmental reactants
        env_reactant_energies = []
        for reactant in balance.environmental_reactants:
            energy = mp_client.get_formation_energy(reactant.formula)
            if energy is None:
                energy = _get_standard_formation_energy(reactant.formula)
            if energy is not None:
                env_reactant_energies.append((energy, reactant.coefficient))

        # Count atoms in target to convert eV/atom to total eV
        target_parsed = parse_formula(state.problem.target_formula)
        if not target_parsed:
            return None
        target_n_atoms = sum(target_parsed.values())

        # Count atoms in each precursor
        precursor_n_atoms = []
        for precursor in state.precursors:
            parsed = parse_formula(precursor.formula)
            if not parsed:
                return None
            precursor_n_atoms.append(sum(parsed.values()))

        # Compute total energies (convert eV/atom to total eV using coefficients)
        # Products
        h_target = target_formation_energy * target_n_atoms  # eV
        h_byproducts = sum(
            energy * coeff for energy, coeff in byproduct_energies
        )  # Already in eV (not eV/atom for molecules)

        # Reactants
        h_precursors = sum(
            energy * n_atoms * coeff
            for energy, n_atoms, coeff in zip(precursor_energies, precursor_n_atoms, balance.precursor_coefficients)
        )
        h_env_reactants = sum(
            energy * coeff for energy, coeff in env_reactant_energies
        )

        # ΔH = products - reactants
        delta_h = (h_target + h_byproducts) - (h_precursors + h_env_reactants)

        return {
            "delta_h": delta_h,
            "h_target": h_target,
            "h_precursors": h_precursors,
            "h_byproducts": h_byproducts,
            "h_env_reactants": h_env_reactants,
        }

    except Exception as e:
        # Calculation failed, return None
        return None


def _get_standard_formation_energy(formula: str) -> float | None:
    """
    Get standard formation energy for common molecules.

    Returns formation energy in eV (total, not per atom).
    Values from NIST Chemistry WebBook.
    """
    # Standard formation energies at 298 K (converted from kJ/mol to eV)
    STANDARD_FORMATION_ENERGIES = {
        "H2O": -2.5,    # -241.8 kJ/mol ≈ -2.5 eV
        "CO2": -4.1,    # -393.5 kJ/mol ≈ -4.1 eV
        "NH3": -0.48,   # -46.1 kJ/mol ≈ -0.48 eV
        "NO2": 0.35,    # +33.2 kJ/mol ≈ +0.35 eV
        "NO": 0.94,     # +90.3 kJ/mol ≈ +0.94 eV
        "N2": 0.0,      # Element reference state
        "O2": 0.0,      # Element reference state
        "H2": 0.0,      # Element reference state
        "Cl2": 0.0,     # Element reference state
        "HCl": -0.96,   # -92.3 kJ/mol ≈ -0.96 eV
        "SO2": -3.1,    # -296.8 kJ/mol ≈ -3.1 eV
    }

    return STANDARD_FORMATION_ENERGIES.get(formula)


def analyze_thermodynamics(state: PlanningState, balance: ReactionBalanceResult, redox: RedoxAnalysisResult, mp_client=None) -> ThermoAnalysisResult:
    notes = []
    if not balance.feasible:
        return ThermoAnalysisResult(
            score=0.0,
            gas_release_moles=0.0,
            gas_uptake_moles=0.0,
            byproduct_count=0,
            decomposition_match=0.0,
            redox_match=0.0,
            notes=("No thermodynamic proxy score was assigned because the route could not be balanced.",),
        )

    gas_release = sum(species.coefficient for species in balance.byproducts if species.formula in GAS_FORMULAS)
    gas_uptake = sum(species.coefficient for species in balance.environmental_reactants if species.formula in GAS_FORMULAS)
    byproduct_count = len(balance.byproducts)

    decomposition_match = 0.25
    classes = {precursor.class_name for precursor in state.precursors}
    if state.target_class == "oxide" and classes & DECOMPOSABLE_OXIDE_PRECURSORS:
        decomposition_match += 0.35
        notes.append("Decomposable salt precursors create a plausible thermodynamic sink for oxide formation.")
    elif state.problem.modality in {"hydrothermal", "precipitation"} and classes & SOLUTION_FRIENDLY_PRECURSORS:
        decomposition_match += 0.25
        notes.append("Solution-friendly precursors support dissolved-ion transport before product nucleation.")
    elif classes & {"oxide", "sulfide", "nitride"}:
        decomposition_match += 0.1

    if gas_release > 0:
        decomposition_match += min(0.25, 0.08 * gas_release)
        notes.append("The balanced route releases gaseous byproducts, which can help pull the reaction forward.")

    redox_match = 0.5
    if redox.required_direction == "none":
        redox_match = 0.8
    elif redox.environment_support == "supported":
        redox_match = 0.85
        notes.append("The reaction environment is consistent with the inferred redox demand.")
    elif redox.environment_support == "unsupported":
        redox_match = 0.1
        notes.append("The reaction environment does not support the inferred redox demand.")

    score = 0.25
    score += 0.25 if balance.feasible else 0.0
    score += min(0.2, 0.08 * gas_release)
    score += 0.2 * min(1.0, decomposition_match)
    score += 0.2 * redox_match
    if gas_uptake > 1.0:
        score -= min(0.1, 0.03 * gas_uptake)
        notes.append("The route depends on a nontrivial amount of environmental reactant uptake.")
    if byproduct_count > 3:
        score -= 0.05
        notes.append("A large byproduct manifold suggests a less clean reaction channel.")

    # Add Materials Project thermodynamic data if available
    hull_energy = None
    formation_energy = None
    decomposition_energy_mp = None
    competing_phases = ()
    is_stable = None
    reaction_driving_force = None
    is_exothermic = None

    if mp_client and mp_client.enabled:
        try:
            mp_data = mp_client.get_thermodynamic_data(state.problem.target_formula)
            if mp_data:
                hull_energy = mp_data.hull_energy_ev_per_atom
                formation_energy = mp_data.formation_energy_ev_per_atom
                decomposition_energy_mp = mp_data.decomposition_energy_ev_per_atom
                competing_phases = mp_data.competing_phases
                is_stable = mp_data.is_stable

                # Adjust score based on hull energy
                if hull_energy is not None:
                    if hull_energy > 0.1:  # Significantly unstable
                        score *= 0.5
                        notes.append(f"Target is {hull_energy:.3f} eV/atom above hull (thermodynamically unstable).")
                    elif hull_energy > 0.0:
                        score *= 0.8
                        notes.append(f"Target is {hull_energy:.3f} eV/atom above hull (metastable).")
                    else:
                        score *= 1.1  # Bonus for stable target
                        notes.append("Target is on the convex hull (thermodynamically stable).")

                # Compute reaction driving force if formation energies available
                if formation_energy is not None and balance.feasible:
                    driving_force_result = _compute_reaction_driving_force(
                        state, balance, formation_energy, mp_client
                    )
                    if driving_force_result is not None:
                        reaction_driving_force = driving_force_result["delta_h"]
                        is_exothermic = reaction_driving_force < 0

                        if is_exothermic:
                            score *= 1.05
                            notes.append(f"Reaction is exothermic (ΔH = {reaction_driving_force:.2f} eV).")
                        else:
                            # Endothermic reactions may need higher temperatures
                            score *= 0.95
                            notes.append(f"Reaction is endothermic (ΔH = {reaction_driving_force:.2f} eV), may require elevated temperatures.")

                # Warn about competing phases
                if competing_phases:
                    score *= 0.9
                    notes.append(f"Competing stable phases in this system: {', '.join(competing_phases[:3])}.")

        except Exception as e:
            notes.append(f"Materials Project lookup failed: {str(e)[:50]}")

    return ThermoAnalysisResult(
        score=max(0.0, min(score, 1.0)),
        gas_release_moles=gas_release,
        gas_uptake_moles=gas_uptake,
        byproduct_count=byproduct_count,
        decomposition_match=max(0.0, min(decomposition_match, 1.0)),
        redox_match=max(0.0, min(redox_match, 1.0)),
        notes=tuple(dict.fromkeys(notes)) or ("The route is thermodynamically plausible under coarse offline proxies.",),
        hull_energy_ev_per_atom=hull_energy,
        formation_energy_ev_per_atom=formation_energy,
        decomposition_energy_ev_per_atom=decomposition_energy_mp,
        competing_phases=competing_phases,
        is_stable=is_stable,
        reaction_driving_force_ev=reaction_driving_force,
        is_exothermic=is_exothermic,
    )


def _solve_volatile_residual(
    state: PlanningState, residual: dict[str, Fraction]
) -> tuple[list[tuple[str, Fraction]] | None, list[tuple[str, Fraction]] | None]:
    if not residual:
        return [], []

    elements = sorted(residual)
    target = [residual[element] for element in elements]
    candidates = []
    for formula in COMMON_BYPRODUCT_FORMULAS:
        counts = _fractional_counts(formula)
        if not set(counts).issubset(set(elements)):
            continue
        vector = tuple(counts.get(element, Fraction(0, 1)) for element in elements)
        if any(vector):
            candidates.append(_CandidateSpecies(formula=formula, role="product", vector=vector))
    for formula in _allowed_environmental_reactants(state):
        counts = _fractional_counts(formula)
        if not set(counts).issubset(set(elements)):
            continue
        vector = tuple(-counts.get(element, Fraction(0, 1)) for element in elements)
        if any(vector):
            candidates.append(_CandidateSpecies(formula=formula, role="reactant", vector=vector))

    solution = _solve_species_mixture(candidates, target)
    if solution is None:
        return None, None

    products = [(candidate.formula, coefficient) for candidate, coefficient in solution if candidate.role == "product"]
    reactants = [(candidate.formula, coefficient) for candidate, coefficient in solution if candidate.role == "reactant"]
    return products, reactants


def _solve_species_mixture(
    candidates: list[_CandidateSpecies], target: list[Fraction]
) -> list[tuple[_CandidateSpecies, Fraction]] | None:
    if not candidates:
        return None

    matrix = [[candidate.vector[row] for candidate in candidates] for row in range(len(target))]
    max_support = min(len(candidates), len(target) + 1)
    for size in range(1, max_support + 1):
        for subset in combinations(range(len(candidates)), size):
            submatrix = [[matrix[row][index] for index in subset] for row in range(len(target))]
            subsolution = _solve_linear_system(submatrix, target)
            if subsolution is None or any(value < 0 for value in subsolution):
                continue
            if any(value == 0 for value in subsolution):
                continue
            if not _matches_target(submatrix, subsolution, target):
                continue
            return [(candidates[index], value) for index, value in zip(subset, subsolution)]
    return None


def _allowed_environmental_reactants(state: PlanningState) -> tuple[str, ...]:
    allowed = set()
    if state.problem.modality in {"hydrothermal", "precipitation"} or state.solvents:
        allowed.add("H2O")
    if _has_oxidizing_atmosphere(state) or not _has_explicit_atmosphere(state):
        allowed.add("O2")
    if _has_reducing_atmosphere(state):
        allowed.add("H2")
    if any(token in {"nh3", "ammonia"} for token in _atmosphere_tokens(state)) or any(
        "ammon" in solvent.lower() for solvent in state.solvents
    ):
        allowed.add("NH3")
    if any(token in INERT_ATMOSPHERES for token in _atmosphere_tokens(state)):
        allowed.add("N2")
    return tuple(sorted(allowed))


def _redox_focus_elements(target_formula: str, target_class: str) -> set[str]:
    elements = set(parse_formula(target_formula))
    if target_class == "oxide":
        return {element for element in elements if element not in {"O", "H"}}
    if target_class == "phosphate":
        return {element for element in elements if element not in {"O", "H"}}
    if target_class == "sulfide":
        return {element for element in elements if element not in {"S", "H"}}
    if target_class == "nitride":
        return {element for element in elements if element not in {"N", "H"}}
    if target_class == "halide":
        return {element for element in elements if element not in {"F", "Cl", "Br", "I", "H"}}
    return {element for element in elements if element not in {"H"}}


def _infer_focus_charge_sum(formula: str, focus_elements: set[str], class_name: str | None = None) -> Fraction | None:
    counts = _fractional_counts(formula)
    known_charge = Fraction(0, 1)
    for element, amount in counts.items():
        if element in focus_elements:
            continue
        state = _typical_oxidation_state(element, counts, class_name)
        if state is None:
            return None
        known_charge += amount * Fraction(state, 1)
    focus_charge = -known_charge
    if sum(count for element, count in counts.items() if element in focus_elements) == 0:
        return None
    return focus_charge


def _typical_oxidation_state(element: str, counts: dict[str, Fraction], class_name: str | None = None) -> int | None:
    oxygen = counts.get("O", Fraction(0, 1))
    hydrogen = counts.get("H", Fraction(0, 1))
    if element == "O":
        return -2
    if element == "H":
        return 1
    if element in ALKALI_METALS:
        return 1
    if element in ALKALINE_EARTH_METALS:
        return 2
    if element in FIXED_POSITIVE_STATES:
        return FIXED_POSITIVE_STATES[element]
    if element == "F":
        return -1
    if element in {"Cl", "Br", "I"}:
        if oxygen > 0 and class_name not in {"chloride", "bromide", "iodide", "halide"}:
            return None
        return -1
    if element == "C":
        if oxygen >= 2 * counts.get("C", Fraction(0, 1)):
            return 4
        if hydrogen and oxygen == 0:
            return -4
        return None
    if element == "N":
        if oxygen > 0:
            return 5
        if hydrogen >= 4 * counts.get("N", Fraction(0, 1)):
            return -3
        return None
    if element == "P":
        return 5 if oxygen > 0 else -3
    if element == "S":
        return 6 if oxygen > 0 else -2
    if element == "B":
        return 3
    if element == "Si":
        return 4
    return None


def _supports_oxidation(state: PlanningState, balance: ReactionBalanceResult) -> bool:
    if any(species.formula == "O2" and species.coefficient > 0 for species in balance.environmental_reactants):
        return True
    if _has_oxidizing_atmosphere(state) or not _has_explicit_atmosphere(state):
        return True
    return any(precursor.class_name == "nitrate" for precursor in state.precursors)


def _supports_reduction(state: PlanningState, balance: ReactionBalanceResult) -> bool:
    if any(species.formula in {"H2", "NH3"} and species.coefficient > 0 for species in balance.environmental_reactants):
        return True
    if _has_reducing_atmosphere(state):
        return True
    if state.target_class in {"sulfide", "nitride"} and not _has_oxidizing_atmosphere(state):
        return True
    return False


def _has_explicit_atmosphere(state: PlanningState) -> bool:
    return any(token in RECOGNIZED_ATMOSPHERES for token in _atmosphere_tokens(state))


def _has_oxidizing_atmosphere(state: PlanningState) -> bool:
    return any(token in OXIDIZING_ATMOSPHERES for token in _atmosphere_tokens(state))


def _has_reducing_atmosphere(state: PlanningState) -> bool:
    return any(token in REDUCING_ATMOSPHERES for token in _atmosphere_tokens(state))


def _atmosphere_tokens(state: PlanningState) -> set[str]:
    tokens = set()
    for operation in state.operations:
        if not operation.atmosphere:
            continue
        normalized = operation.atmosphere.lower().replace("/", ",")
        tokens.update(part.strip() for part in normalized.split(",") if part.strip())
    return tokens


def _unused_precursors(state: PlanningState, coefficients: list[Fraction]) -> tuple[str, ...]:
    unused = []
    for precursor, coefficient in zip(state.precursors, coefficients):
        if coefficient == 0:
            unused.append(precursor.formula)
    return tuple(unused)


def _coverage_fraction(state: PlanningState, framework_elements: list[str]) -> float:
    covered = {element for precursor in state.precursors for element in precursor.elements}
    required = set(framework_elements)
    return len(required & covered) / max(1, len(required))


def _combine_species(species, coefficients: list[Fraction]) -> dict[str, Fraction]:
    totals: dict[str, Fraction] = {}
    for record, coefficient in zip(species, coefficients):
        if coefficient == 0:
            continue
        for element, amount in _fractional_counts(record.formula).items():
            totals[element] = totals.get(element, Fraction(0, 1)) + coefficient * amount
    return totals


def _fractional_counts(formula: str) -> dict[str, Fraction]:
    return {element: _to_fraction(amount) for element, amount in parse_formula(formula).items()}


def _solve_basic_feasible(matrix: list[list[Fraction]], rhs: list[Fraction]) -> list[Fraction] | None:
    if not matrix or not matrix[0]:
        return None
    num_variables = len(matrix[0])
    max_support = min(num_variables, len(rhs))
    for size in range(max_support, 0, -1):
        for subset in combinations(range(num_variables), size):
            submatrix = [[row[index] for index in subset] for row in matrix]
            subsolution = _solve_linear_system(submatrix, rhs)
            if subsolution is None or any(value < 0 for value in subsolution):
                continue
            if any(value == 0 for value in subsolution):
                continue
            if not _matches_target(submatrix, subsolution, rhs):
                continue
            solution = [Fraction(0, 1) for _ in range(num_variables)]
            for index, value in zip(subset, subsolution):
                solution[index] = value
            return solution
    return None


def _solve_linear_system(matrix: list[list[Fraction]], rhs: list[Fraction]) -> list[Fraction] | None:
    if not matrix:
        return []
    row_count = len(matrix)
    col_count = len(matrix[0])
    augmented = [list(row) + [rhs[index]] for index, row in enumerate(matrix)]
    pivot_columns = []
    pivot_row = 0

    for column in range(col_count):
        candidate = None
        for row in range(pivot_row, row_count):
            if augmented[row][column] != 0:
                candidate = row
                break
        if candidate is None:
            continue
        augmented[pivot_row], augmented[candidate] = augmented[candidate], augmented[pivot_row]
        pivot_value = augmented[pivot_row][column]
        augmented[pivot_row] = [value / pivot_value for value in augmented[pivot_row]]
        for row in range(row_count):
            if row == pivot_row or augmented[row][column] == 0:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                current - factor * pivot
                for current, pivot in zip(augmented[row], augmented[pivot_row])
            ]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == row_count:
            break

    for row in augmented:
        if all(value == 0 for value in row[:-1]) and row[-1] != 0:
            return None

    solution = [Fraction(0, 1) for _ in range(col_count)]
    for row_index, column in enumerate(pivot_columns):
        solution[column] = augmented[row_index][-1]
    return solution


def _matches_target(matrix: list[list[Fraction]], solution: list[Fraction], rhs: list[Fraction]) -> bool:
    for row, target in zip(matrix, rhs):
        total = sum(value * coefficient for value, coefficient in zip(row, solution))
        if total != target:
            return False
    return True


def _format_equation(
    state: PlanningState,
    precursor_coefficients: list[Fraction],
    reactants: list[tuple[str, Fraction]],
    products: list[tuple[str, Fraction]],
) -> str:
    left = []
    for precursor, coefficient in zip(state.precursors, precursor_coefficients):
        if coefficient > 0:
            left.append(f"{_format_coefficient(coefficient)} {precursor.formula}")
    for formula, coefficient in reactants:
        if coefficient > 0:
            left.append(f"{_format_coefficient(coefficient)} {formula}")

    right = [state.problem.target_formula]
    for formula, coefficient in products:
        if coefficient > 0:
            right.append(f"{_format_coefficient(coefficient)} {formula}")
    return " + ".join(left) + " -> " + " + ".join(right)


def _format_coefficient(value: Fraction) -> str:
    if value == 1:
        return "1"
    if value.denominator <= 12:
        return f"{value.numerator}/{value.denominator}" if value.denominator != 1 else str(value.numerator)
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def _to_fraction(value: float | int | Fraction) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value, 1)
    return Fraction(str(value))
