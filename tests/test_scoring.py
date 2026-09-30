from synthesis_planner.core.formula import infer_target_class, parse_formula
from synthesis_planner.core.schema import PlanningProblem, PlanningState, PrecursorRecord
from synthesis_planner.core.scoring import evaluate_state


def test_evaluate_state_rewards_complete_oxide_route(processed_data):
    problem = PlanningProblem(target_formula="BaTiO3")
    state = PlanningState(
        problem=problem,
        target_elements=tuple(sorted(parse_formula(problem.target_formula))),
        target_class=infer_target_class(problem.target_formula),
        stage="terminal",
        precursors=(
            PrecursorRecord("BaCO3", "carbonate", ("Ba", "C", "O")),
            PrecursorRecord("TiO2", "oxide", ("Ti", "O")),
        ),
        operations=(),
    )
    route = evaluate_state(state, [])
    assert route.score.stoich == 1.0
    assert route.score.precursor > 0.0
    assert route.thermo.score > 0.0
    assert route.hard_checks.reaction_balance is not None
    assert route.hard_checks.reaction_balance.feasible


def test_solved_to_stock_dominates_terminal_reward(processed_data):
    from synthesis_planner.core.schema import EvaluationConfig
    from synthesis_planner.data.stock import Stock

    problem = PlanningProblem(target_formula="BaTiO3")
    state = PlanningState(
        problem=problem,
        target_elements=tuple(sorted(parse_formula(problem.target_formula))),
        target_class=infer_target_class(problem.target_formula),
        stage="terminal",
        precursors=(
            PrecursorRecord("BaCO3", "carbonate", ("Ba", "C", "O")),
            PrecursorRecord("TiO2", "oxide", ("Ti", "O")),
        ),
        operations=(),
    )
    stock = Stock(formulas=frozenset({"BaCO3", "TiO2"}))
    solved = evaluate_state(state, [], EvaluationConfig(stock=stock))
    no_stock = evaluate_state(state, [], EvaluationConfig())

    # A fully-in-stock route earns the maximal solved-to-stock term.
    assert solved.score.stock == 1.0
    # ...and that term dominates: the solved reward is materially higher than
    # the same route scored without a stock context.
    assert solved.score.total > no_stock.score.total + 2.0

    # Half-in-stock route earns partial credit only.
    half_state = PlanningState(
        problem=problem,
        target_elements=state.target_elements,
        target_class=state.target_class,
        stage="terminal",
        precursors=(
            PrecursorRecord("BaTiO3", "oxide", ("Ba", "Ti", "O")),  # not a commodity
            PrecursorRecord("TiO2", "oxide", ("Ti", "O")),
        ),
        operations=(),
    )
    half = evaluate_state(half_state, [], EvaluationConfig(stock=stock))
    assert half.score.stock == 0.5
    assert half.score.total < solved.score.total


def test_retrieval_term_is_demoted(processed_data):
    from synthesis_planner.core.schema import EvaluationConfig
    from synthesis_planner.core.formula import safe_infer_target_class
    from synthesis_planner.data.datasets import load_processed_routes
    from synthesis_planner.data.retrieval import RetrievalIndex

    routes = load_processed_routes(processed_data, "solid_state")
    retrieval = RetrievalIndex(routes)
    analogs = retrieval.retrieve("BaTiO3", top_k=12)
    problem = PlanningProblem(target_formula="BaTiO3")
    state = PlanningState(
        problem=problem,
        target_elements=tuple(sorted(parse_formula("BaTiO3"))),
        target_class=safe_infer_target_class("BaTiO3"),
        stage="terminal",
        precursors=(
            PrecursorRecord("BaCO3", "carbonate", ("Ba", "C", "O")),
            PrecursorRecord("TiO2", "oxide", ("Ti", "O")),
        ),
        operations=(),
    )
    route = evaluate_state(state, analogs, EvaluationConfig())
    # Retrieval now contributes at most retrieval_weight (0.2) * retrieval_score,
    # so it cannot dominate or manufacture a high leaf reward on its own.
    assert route.score.retrieval >= 0.0
    contribution = EvaluationConfig().retrieval_weight * route.score.retrieval
    assert contribution <= 0.2


def test_judge_flags_low_temperature_decomposition_risk():
    from synthesis_planner.core.schema import NumericRange, OperationRecord

    problem = PlanningProblem(target_formula="BaTiO3")
    state = PlanningState(
        problem=problem,
        target_elements=tuple(sorted(parse_formula(problem.target_formula))),
        target_class=infer_target_class(problem.target_formula),
        stage="terminal",
        precursors=(
            PrecursorRecord("BaCO3", "carbonate", ("Ba", "C", "O")),
            PrecursorRecord("TiO2", "oxide", ("Ti", "O")),
        ),
        operations=(
            OperationRecord("mix", source_label="mix"),
            OperationRecord("heat", temperature_c=NumericRange(500.0, 500.0, "C"), source_label="calcine"),
        ),
    )
    route = evaluate_state(state, [])
    assert "decomposition_risk" in route.judge.flags
