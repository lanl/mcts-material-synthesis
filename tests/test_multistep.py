"""Tests for the multi-stage thermal MDP (#1) and solution-intermediate DAGs (#2)."""

from synthesis_planner.core.grammar import apply_action, expand_state
from synthesis_planner.core.retro_planner import insert_solution_intermediate
from synthesis_planner.core.schema import (
    DAGNode,
    HardCheckResult,
    JudgeResult,
    NumericRange,
    OperationRecord,
    PlannedRoute,
    PlanningProblem,
    PlanningState,
    PrecursorRecord,
    ScoreBreakdown,
    ThermoAnalysisResult,
)


# ---- #1: multi-stage thermal MDP ------------------------------------------------

def _heating_state(operations):
    return PlanningState(
        problem=PlanningProblem(target_formula="BaTiO3", modality="solid_state"),
        target_elements=("Ba", "Ti", "O"),
        target_class="oxide",
        stage="heating",
        precursors=(PrecursorRecord("BaCO3", "carbonate", ("Ba", "C", "O")),),
        operations=operations,
    )


def test_first_heating_node_offers_calcine_but_not_finish():
    """With no firing yet, the search can add a stage but not stop (no bare no-heat route)."""
    actions = expand_state(_heating_state(()), analogs=[], candidate_precursor_sets=[])
    kinds = {a.kind for a in actions}
    assert "add_heat_stage" in kinds
    assert "finish_heating" not in kinds  # cannot finish before any firing


def test_finish_available_after_one_stage():
    state = _heating_state((OperationRecord("heat", NumericRange(900.0, 900.0, "C")),))
    actions = expand_state(state, analogs=[], candidate_precursor_sets=[])
    assert any(a.kind == "finish_heating" for a in actions)
    # Subsequent stages prepend an intergrind (regrind) before re-firing.
    add = [a for a in actions if a.kind == "add_heat_stage"]
    assert add and any(op.verb == "mix" for op in add[0].payload)


def test_heating_stage_cap_forces_stop():
    """At the 3-stage cap the only legal action is to stop."""
    three = tuple(OperationRecord("heat", NumericRange(900.0, 900.0, "C")) for _ in range(3))
    actions = expand_state(_heating_state(three), analogs=[], candidate_precursor_sets=[])
    assert [a.kind for a in actions] == ["finish_heating"]


def test_add_heat_stage_stays_in_heating_and_appends():
    state = _heating_state(())
    add = next(a for a in expand_state(state, [], []) if a.kind == "add_heat_stage")
    nxt = apply_action(state, add, analogs=[])
    assert nxt.stage == "heating"  # remains in the MDP for another decision
    assert sum(1 for op in nxt.operations if op.verb == "heat") == 1


# ---- #2: solution-route intermediate decomposition -----------------------------

def _route(operations):
    return PlannedRoute(
        target_formula="CoFe2O4",
        modality="precipitation",
        precursors=(
            PrecursorRecord("Co(NO3)2", "nitrate", ("Co", "N", "O")),
            PrecursorRecord("Fe(NO3)3", "nitrate", ("Fe", "N", "O")),
        ),
        solvents=("water",),
        operations=operations,
        evidence_dois=(),
        analog_targets=(),
        hard_checks=HardCheckResult(True, (), (), 1.0, ()),
        score=ScoreBreakdown(1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1),
        thermo=ThermoAnalysisResult(1.0, 0, 0, 0, 1.0, 1.0, ()),
        judge=JudgeResult(1.0, (), ()),
        mcts_value=1.0,
    )


def _node(operations, modality="precipitation"):
    return DAGNode(
        target_formula="CoFe2O4", modality=modality, depth=0,
        recipe=_route(operations),
    )


def test_precipitate_then_calcine_splits_into_intermediate():
    ops = (
        OperationRecord("mix", source_label="dissolve"),
        OperationRecord("precipitate", source_label="precipitate"),
        OperationRecord("wash"),
        OperationRecord("dry"),
        OperationRecord("heat", NumericRange(500.0, 500.0, "C"), source_label="calcine"),
    )
    node = insert_solution_intermediate(_node(ops), is_in_stock=lambda p: True)
    assert len(node.children) == 1
    interm = node.children[0]
    assert interm.is_intermediate
    # Intermediate carries the precursor-forming ops; salts are its stock leaves.
    assert [op.verb for op in interm.operations] == ["mix", "precipitate", "wash", "dry"]
    assert len(interm.children) == 2 and all(c.in_stock for c in interm.children)
    # Parent keeps only the calcination.
    assert [op.verb for op in node.operations] == ["heat"]


def test_no_calcination_is_left_flat():
    """A route that ends at wash->dry (no calcine) is not split."""
    ops = (OperationRecord("precipitate"), OperationRecord("wash"), OperationRecord("dry"))
    node = insert_solution_intermediate(_node(ops), is_in_stock=lambda p: True)
    assert not any(c.is_intermediate for c in node.children)


def test_hydrothermal_hold_is_not_split():
    """A hold (heat before wash/dry) is a direct crystallisation, not a calcination."""
    ops = (
        OperationRecord("mix", source_label="dissolve"),
        OperationRecord("heat", NumericRange(180.0, 180.0, "C"), source_label="hydrothermal hold"),
        OperationRecord("wash"),
        OperationRecord("dry"),
    )
    node = insert_solution_intermediate(_node(ops, modality="hydrothermal"), is_in_stock=lambda p: True)
    assert not any(c.is_intermediate for c in node.children)
