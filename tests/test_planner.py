from synthesis_planner.planner import SynthesisPlanner
from synthesis_planner.schema import PlanningProblem


def test_planner_generates_ranked_routes(sample_raw_data, processed_data):
    planner = SynthesisPlanner(sample_raw_data, processed_data)
    routes = planner.plan(
        PlanningProblem(target_formula="BaTiO3"),
        iterations=25,
        top_k=3,
        rollout_count=3,
        seed=7,
    )
    assert routes
    assert routes[0].target_formula == "BaTiO3"
    assert any(precursor.formula == "BaCO3" for precursor in routes[0].precursors)
    assert routes[0].score.total > 0.0


def test_planner_generates_hydrothermal_routes(sample_raw_data, processed_data):
    planner = SynthesisPlanner(sample_raw_data, processed_data)
    routes = planner.plan(
        PlanningProblem(target_formula="BaTiO3", modality="hydrothermal"),
        iterations=10,
        top_k=1,
        rollout_count=2,
        seed=4,
    )
    assert routes
    assert routes[0].modality == "hydrothermal"
    assert routes[0].solvents


def test_planner_generates_precipitation_routes(sample_raw_data, processed_data):
    planner = SynthesisPlanner(sample_raw_data, processed_data)
    routes = planner.plan(
        PlanningProblem(target_formula="TiO2", modality="precipitation"),
        iterations=10,
        top_k=1,
        rollout_count=2,
        seed=4,
    )
    assert routes
    assert routes[0].modality == "precipitation"
    assert any(operation.verb in {"precipitate", "wash"} for operation in routes[0].operations)
