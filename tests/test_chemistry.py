from synthesis_planner.chemistry import analyze_redox, analyze_thermodynamics, balance_route
from synthesis_planner.formula import infer_target_class, parse_formula
from synthesis_planner.schema import NumericRange, OperationRecord, PlanningProblem, PlanningState, PrecursorRecord


def _state_for(target_formula="BaTiO3", precursors=(), operations=(), modality="solid_state"):
    problem = PlanningProblem(target_formula=target_formula, modality=modality)
    return PlanningState(
        problem=problem,
        target_elements=tuple(sorted(parse_formula(problem.target_formula))),
        target_class=infer_target_class(problem.target_formula),
        stage="terminal",
        precursors=precursors,
        operations=operations,
    )


def test_balance_route_finds_exact_batio3_reaction():
    state = _state_for(
        precursors=(
            PrecursorRecord("BaCO3", "carbonate", ("Ba", "C", "O")),
            PrecursorRecord("TiO2", "oxide", ("Ti", "O")),
        ),
        operations=(
            OperationRecord("mix", source_label="mix"),
            OperationRecord("heat", temperature_c=NumericRange(900.0, 900.0, "C"), source_label="calcine"),
        ),
    )

    balance = balance_route(state)
    assert balance.feasible
    assert balance.precursor_coefficients == (1.0, 1.0)
    assert any(species.formula == "CO2" for species in balance.byproducts)


def test_balance_route_does_not_invent_unavailable_elements_in_byproducts():
    state = _state_for(
        precursors=(
            PrecursorRecord("Ba(NO3)2", "nitrate", ("Ba", "N", "O")),
            PrecursorRecord("TiO2", "oxide", ("Ti", "O")),
        ),
        operations=(
            OperationRecord("mix", source_label="mix"),
            OperationRecord("heat", temperature_c=NumericRange(900.0, 900.0, "C"), source_label="calcine"),
        ),
    )

    balance = balance_route(state)
    assert balance.feasible
    assert all(species.formula != "CO2" for species in balance.byproducts)


def test_balance_route_rejects_nonvolatile_extra_elements():
    state = _state_for(
        precursors=(
            PrecursorRecord("BaCO3", "carbonate", ("Ba", "C", "O")),
            PrecursorRecord("Na2TiO3", "oxide", ("Na", "Ti", "O")),
        ),
        operations=(
            OperationRecord("mix", source_label="mix"),
            OperationRecord("heat", temperature_c=NumericRange(900.0, 900.0, "C"), source_label="calcine"),
        ),
    )

    balance = balance_route(state)
    assert not balance.feasible
    assert "Na" in balance.residual_elements


def test_redox_analysis_flags_reduction_in_air():
    state = _state_for(
        target_formula="FeO",
        precursors=(PrecursorRecord("Fe2O3", "oxide", ("Fe", "O")),),
        operations=(
            OperationRecord("mix", source_label="mix"),
            OperationRecord(
                "heat",
                temperature_c=NumericRange(700.0, 700.0, "C"),
                atmosphere="air",
                source_label="anneal",
            ),
        ),
    )

    balance = balance_route(state)
    redox = analyze_redox(state, balance)
    assert balance.feasible
    assert redox.required_direction == "reduction"
    assert "oxidizing_atmosphere_mismatch" in redox.flags


def test_unrecognized_atmosphere_label_does_not_block_ambient_oxidation():
    state = _state_for(
        target_formula="Fe2O3",
        precursors=(PrecursorRecord("FeO", "oxide", ("Fe", "O")),),
        operations=(
            OperationRecord("mix", source_label="mix"),
            OperationRecord(
                "heat",
                temperature_c=NumericRange(700.0, 700.0, "C"),
                atmosphere="alumina",
                source_label="anneal",
            ),
        ),
    )

    balance = balance_route(state)
    redox = analyze_redox(state, balance)
    assert balance.feasible
    assert redox.required_direction == "oxidation"
    assert redox.environment_support == "supported"


def test_thermo_features_reward_gas_releasing_oxide_route():
    carbonate_state = _state_for(
        precursors=(
            PrecursorRecord("BaCO3", "carbonate", ("Ba", "C", "O")),
            PrecursorRecord("TiO2", "oxide", ("Ti", "O")),
        ),
        operations=(
            OperationRecord("mix", source_label="mix"),
            OperationRecord("heat", temperature_c=NumericRange(900.0, 900.0, "C"), source_label="calcine"),
        ),
    )
    oxide_state = _state_for(
        precursors=(
            PrecursorRecord("BaO", "oxide", ("Ba", "O")),
            PrecursorRecord("TiO2", "oxide", ("Ti", "O")),
        ),
        operations=(
            OperationRecord("mix", source_label="mix"),
            OperationRecord("heat", temperature_c=NumericRange(900.0, 900.0, "C"), source_label="calcine"),
        ),
    )

    carbonate_balance = balance_route(carbonate_state)
    oxide_balance = balance_route(oxide_state)
    carbonate_redox = analyze_redox(carbonate_state, carbonate_balance)
    oxide_redox = analyze_redox(oxide_state, oxide_balance)
    carbonate_thermo = analyze_thermodynamics(carbonate_state, carbonate_balance, carbonate_redox)
    oxide_thermo = analyze_thermodynamics(oxide_state, oxide_balance, oxide_redox)

    assert carbonate_thermo.gas_release_moles > oxide_thermo.gas_release_moles
    assert carbonate_thermo.score > oxide_thermo.score
