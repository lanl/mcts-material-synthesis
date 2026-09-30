"""Tests for the DAG recursion driver (WS-E)."""

from synthesis_planner.core.retro_planner import plan_retro
from synthesis_planner.core.schema import (
    HardCheckResult,
    JudgeResult,
    PlannedRoute,
    PrecursorRecord,
    ScoreBreakdown,
    ThermoAnalysisResult,
)


def _recipe(target, precursor_formulas, valid=True):
    precursors = tuple(
        PrecursorRecord(f, "oxide", tuple(c for c in f if c.isupper())) for f in precursor_formulas
    )
    return PlannedRoute(
        target_formula=target,
        modality="solid_state",
        precursors=precursors,
        solvents=(),
        operations=(),
        evidence_dois=(),
        analog_targets=(),
        hard_checks=HardCheckResult(valid=valid, flags=(), notes=(), coverage_fraction=1.0, blocking_flags=()),
        score=ScoreBreakdown(1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 5.0),
        thermo=ThermoAnalysisResult(1.0, 0, 0, 0, 1.0, 1.0, ()),
        judge=JudgeResult(1.0, (), ()),
        mcts_value=5.0,
    )


# A tiny synthetic world: BaTiO3 <- BaCO3 (stock) + TiO2 (stock) is depth-1.
# BaZrO3 <- BaCO3 (stock) + BaTiO3 (recursive sub-target) forces depth-2, and
# BaTiO3 then decomposes into two stock precursors.
STOCK = {"BaCO3", "TiO2", "SrCO3", "ZrO2"}
KNOWN_TARGETS = {"BaTiO3", "BaZrO3", "SrTiO3"}

RECIPES = {
    "BaTiO3": _recipe("BaTiO3", ["BaCO3", "TiO2"]),
    "BaZrO3": _recipe("BaZrO3", ["BaTiO3", "ZrO2"]),  # BaTiO3 is non-stock -> recurse
    "SrTiO3": _recipe("SrTiO3", ["SrCO3", "TiO2"]),
}


def _plan_single(formula, modality):
    return RECIPES.get(formula)


def _in_stock(precursor):
    formula = precursor if isinstance(precursor, str) else precursor.formula
    return formula in STOCK


def _is_recursion_candidate(formula):
    return formula in KNOWN_TARGETS and formula not in STOCK


def test_depth_1_solved():
    dag = plan_retro("BaTiO3", "solid_state", _plan_single, _in_stock, _is_recursion_candidate, max_depth=2)
    assert dag.depth == 1
    assert dag.node_count == 3  # root + 2 stock leaves
    assert dag.is_solved
    leaves = dag.root.leaves()
    assert all(leaf.in_stock for leaf in leaves)


def test_depth_2_dag_assembled_and_solved():
    dag = plan_retro("BaZrO3", "solid_state", _plan_single, _in_stock, _is_recursion_candidate, max_depth=2)
    assert dag.depth == 2
    assert dag.is_solved
    # The BaTiO3 sub-target node has its own recipe + children.
    child_formulas = [c.target_formula for c in dag.root.children]
    assert "BaTiO3" in child_formulas
    batio3 = next(c for c in dag.root.children if c.target_formula == "BaTiO3")
    assert batio3.recipe is not None
    assert {leaf.target_formula for leaf in batio3.leaves()} == {"BaCO3", "TiO2"}


def test_depth_cap_leaves_dangling():
    # With max_depth=0 the root's non-stock precursor cannot be expanded.
    dag = plan_retro("BaZrO3", "solid_state", _plan_single, _in_stock, _is_recursion_candidate, max_depth=0)
    assert not dag.is_solved
    dangling = [leaf for leaf in dag.root.leaves() if leaf.dangling]
    assert any(leaf.target_formula == "BaTiO3" for leaf in dangling)


def test_cycle_guard():
    # A self-referential recipe (X <- X) must not infinitely recurse.
    recipes = {"X": _recipe("X", ["X", "TiO2"])}
    known = {"X"}
    stock = {"TiO2"}
    dag = plan_retro(
        "X",
        "solid_state",
        lambda f, m: recipes.get(f),
        lambda p: (p if isinstance(p, str) else p.formula) in stock,
        lambda f: f in known and f not in stock,
        max_depth=5,
    )
    # X appears as its own precursor -> cycle-guarded into a dangling leaf.
    x_leaves = [n for n in dag.root.leaves() if n.target_formula == "X"]
    assert x_leaves and all(leaf.dangling for leaf in x_leaves)


def test_unsolved_when_dangling_non_stock():
    # A precursor that is neither stock nor a known target -> dangling, unsolved.
    dag = plan_retro(
        "BaTiO3",
        "solid_state",
        lambda f, m: _recipe("BaTiO3", ["BaCO3", "WeirdOxide"]),
        _in_stock,
        _is_recursion_candidate,
        max_depth=2,
    )
    assert not dag.is_solved
