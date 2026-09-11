"""Unit tests for the retro benchmark harness (metrics + leakage audit)."""

from synthesis_planner.core.schema import (
    HardCheckResult,
    JudgeResult,
    NumericRange,
    OperationRecord,
    PlannedRoute,
    PrecursorRecord,
    RouteRecord,
    ScoreBreakdown,
    ThermoAnalysisResult,
)
from synthesis_planner.retro_benchmark import (
    build_gold_targets,
    leakage_audit,
    _recall_metrics,
)


def _gold_route(target, sys, precursors, temp=1000.0):
    return RouteRecord(
        route_id=f"r-{target}-{sys}", source_doi="", publication_year=2020,
        modality="solid_state", target_formula=target,
        target_elements=tuple(sorted(set("".join(precursors)))), chemical_system=sys,
        target_class="oxide",
        precursors=tuple(PrecursorRecord(f, "oxide", (f,)) for f in precursors),
        solvents=(), operations=(OperationRecord("heat", NumericRange(temp, temp, "C")),),
        reaction_string="", paragraph_excerpt="", source_dataset="t",
    )


def _pred(precursors, temp=1000.0):
    return PlannedRoute(
        target_formula="BaTiO3", modality="solid_state",
        precursors=tuple(PrecursorRecord(f, "oxide", (f,)) for f in precursors),
        solvents=(), operations=(OperationRecord("heat", NumericRange(temp, temp, "C")),),
        evidence_dois=(), analog_targets=(),
        hard_checks=HardCheckResult(True, (), (), 1.0, ()),
        score=ScoreBreakdown(1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1),
        thermo=ThermoAnalysisResult(1.0, 0, 0, 0, 1.0, 1.0, ()),
        judge=JudgeResult(1.0, (), ()), mcts_value=1.0,
    )


def test_leakage_audit_detects_clean_and_leaked_splits():
    train = [_gold_route("BaTiO3", "Ba-O-Ti", ["BaO", "TiO2"])]
    test_clean = [_gold_route("SrTiO3", "O-Sr-Ti", ["SrO", "TiO2"])]
    assert leakage_audit(train, test_clean, "chemical_system")["clean"] is True

    test_leaked = [_gold_route("BaTi2O5", "Ba-O-Ti", ["BaO", "TiO2"])]
    report = leakage_audit(train, test_leaked, "chemical_system")
    assert report["clean"] is False and report["n_leaked"] == 1


def test_build_gold_targets_unions_multiple_routes():
    routes = [
        _gold_route("BaTiO3", "Ba-O-Ti", ["BaCO3", "TiO2"], temp=1100.0),
        _gold_route("BaTiO3", "Ba-O-Ti", ["BaO", "TiO2"], temp=900.0),
    ]
    gold = build_gold_targets(routes)
    assert set(gold) == {"BaTiO3"}
    assert len(gold["BaTiO3"].precursor_sets) == 2
    assert sorted(gold["BaTiO3"].temperatures) == [900.0, 1100.0]


def test_recall_metrics_exact_and_jaccard():
    routes = [_gold_route("BaTiO3", "Ba-O-Ti", ["BaCO3", "TiO2"])]
    gold = build_gold_targets(routes)["BaTiO3"]

    # Exact match on the top prediction.
    hit = _recall_metrics([_pred(["BaCO3", "TiO2"])], gold)
    assert hit["recall_at_1"] and hit["recall_at_k"] and hit["jaccard"] == 1.0

    # Partial overlap: one of two precursors correct.
    partial = _recall_metrics([_pred(["BaO", "TiO2"])], gold)
    assert not partial["recall_at_1"]
    assert 0.0 < partial["jaccard"] < 1.0

    # recall@k picks up a correct set lower in the list.
    at_k = _recall_metrics([_pred(["BaO", "TiO2"]), _pred(["BaCO3", "TiO2"])], gold)
    assert not at_k["recall_at_1"] and at_k["recall_at_k"]
