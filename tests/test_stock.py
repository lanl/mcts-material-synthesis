"""Tests for stock mining and the solved-to-stock criterion (WS-A)."""

import json

from synthesis_planner.core.schema import PrecursorRecord, RouteRecord
from synthesis_planner.data.stock import (
    Stock,
    build_stock,
    cation_elements,
    is_simple_compound,
    load_or_build_stock,
    route_is_solved,
    stock_coverage,
)


def _route(target, precursor_formulas):
    precursors = tuple(
        PrecursorRecord(
            formula=f,
            class_name="carbonate" if "CO3" in f else "oxide",
            elements=tuple(c for c in f if c.isupper()),
        )
        for f in precursor_formulas
    )
    return RouteRecord(
        route_id=f"r-{target}",
        source_doi="10.1/x",
        publication_year=2020,
        modality="solid_state",
        target_formula=target,
        target_elements=tuple(c for c in target if c.isupper()),
        chemical_system="x",
        target_class="oxide",
        precursors=precursors,
        solvents=(),
        operations=(),
        reaction_string="",
        paragraph_excerpt="",
        source_dataset="test",
    )


def test_cation_elements_and_simple():
    assert cation_elements("BaCO3") == {"Ba"}
    assert cation_elements("TiO2") == {"Ti"}
    assert cation_elements("BaTiO3") == {"Ba", "Ti"}
    assert is_simple_compound("BaCO3")
    assert is_simple_compound("TiO2")
    assert is_simple_compound("SiO2")
    assert not is_simple_compound("BaTiO3")
    assert not is_simple_compound("ZnFe2O4")


def test_multi_cation_never_in_stock_even_if_frequent():
    # BaTiO3 appears many times but must stay a recursion candidate.
    routes = [_route("X", ["BaTiO3"]) for _ in range(50)]
    stock = build_stock(routes, min_frequency=5)
    assert "BaTiO3" not in stock.formulas
    assert not stock.contains(PrecursorRecord("BaTiO3", "oxide", ("Ba", "Ti", "O")))


def test_simple_commodities_in_stock():
    stock = build_stock([_route("BaTiO3", ["BaCO3", "TiO2"]) for _ in range(30)], min_frequency=5)
    assert stock.contains(PrecursorRecord("BaCO3", "carbonate", ("Ba", "C", "O")))
    assert stock.contains(PrecursorRecord("TiO2", "oxide", ("Ti", "O")))
    # Rule-based inclusion: a simple oxide not in the mined set is still stock.
    assert stock.contains(PrecursorRecord("MgO", "oxide", ("Mg", "O")))
    # Bare-formula membership works for simple compounds.
    assert stock.contains("CaO")


def test_min_frequency_controls_explicit_set_size():
    routes = [_route("T", ["BaCO3", "TiO2"]) for _ in range(30)]
    routes += [_route("U", ["Nb2O5"]) for _ in range(3)]
    low = build_stock(routes, min_frequency=2)
    high = build_stock(routes, min_frequency=10)
    assert "Nb2O5" in low.formulas
    assert "Nb2O5" not in high.formulas
    assert high.size() <= low.size()


def test_route_is_solved_and_coverage():
    stock = Stock(formulas=frozenset({"BaCO3", "TiO2"}))
    ok = (PrecursorRecord("BaCO3", "carbonate", ("Ba",)), PrecursorRecord("TiO2", "oxide", ("Ti",)))
    mixed = (PrecursorRecord("BaTiO3", "oxide", ("Ba", "Ti")), PrecursorRecord("TiO2", "oxide", ("Ti",)))
    assert route_is_solved(ok, stock)
    assert not route_is_solved(mixed, stock)
    assert route_is_solved((), stock) is False
    assert stock_coverage(mixed, stock) == 0.5


def test_load_or_build_caches(tmp_path):
    routes = [_route("BaTiO3", ["BaCO3", "TiO2"]) for _ in range(30)]
    stock = build_stock(routes, min_frequency=5)
    (tmp_path / "stock.json").write_text(json.dumps(stock.to_dict()))
    loaded = load_or_build_stock(tmp_path, min_frequency=5)
    assert loaded.formulas == stock.formulas
    assert loaded.min_frequency == 5
