from synthesis_planner.benchmark import build_split, evaluate_split
from synthesis_planner.datasets import load_processed_routes
from synthesis_planner.planner import SynthesisPlanner


def test_target_formula_split_has_no_target_overlap(processed_data, sample_raw_data):
    routes = load_processed_routes(processed_data, "solid_state")
    train_routes, test_routes = build_split(routes, "target_formula", test_fraction=0.5, seed=1)
    assert {route.target_formula for route in train_routes}.isdisjoint({route.target_formula for route in test_routes})


def test_benchmark_returns_metrics(processed_data, sample_raw_data):
    routes = load_processed_routes(processed_data, "solid_state")
    planner = SynthesisPlanner(sample_raw_data, processed_data)
    train_routes, test_routes = build_split(routes, "random", test_fraction=0.5, seed=1)
    summary = evaluate_split(
        planner,
        train_routes,
        test_routes,
        split_type="random",
        iterations=10,
        top_k=1,
        rollout_count=2,
        seed=3,
    )
    assert summary.n_train >= 1
    assert summary.n_test >= 1
    assert 0.0 <= summary.top1_validity_rate <= 1.0
