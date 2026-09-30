"""Stock (buildable-precursor set) mining and the solved-to-stock criterion.

This is the materials-synthesis analog of the *purchasable building block*
catalog that terminates an organic-retrosynthesis search (Segler 2018;
AiZynthFinder). A synthesis route is **solved** when every leaf precursor is in
the stock, exactly mirroring "a set of purchasable building blocks is reached."

The stock is *mined*, not vendored: we take every precursor formula whose
corpus frequency is at least ``min_frequency`` **and** that is a chemically
*simple* commodity (a single-cation oxide / carbonate / nitrate / hydroxide /
acetate / sulfate / halide / sulfide, or an elemental metal), plus a rule-based
inclusion for any simple commodity of an allowed class regardless of frequency.

Crucially, **multi-cation complex compounds are never in stock** (perovskites,
spinels, garnets, titanates, ...). These are precisely the recursion candidates
that can appear as both a precursor and a target elsewhere in the corpus
(BaTiO3, SrTiO3, YBa2Cu3O7, ZnFe2O4, ...), so keeping them out of stock is what
lets the DAG driver recurse on them.

Stock-size sensitivity to ``min_frequency`` on the combined processed corpora
(19,488 solid-state + 35,675 solution routes, 5,130 distinct precursor
formulas, 127,078 occurrences):

    min_frequency   explicit high-frequency simple formulas
    -------------   -----------------------------------------
        5               ~1150
       10               ~740
       20               ~490   <- default
       50               ~285
      100               ~190

Because the ``contains`` test also admits *any* simple commodity-class compound
via the rule-based path, the effective stock is broader than the explicit
formula set; ``min_frequency`` mainly controls how many rarer simple formulas
are pinned into the cached explicit set and reported as the frequency core.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable

from ..core.formula import safe_element_set
from ..core.schema import PrecursorRecord, RouteRecord

# Elements that only ever act as anions / non-metal framework in these corpora.
# Everything else in a formula is treated as a (metal) cation for the purpose of
# deciding whether a compound is "simple" (single-cation, hence a commodity).
ANION_ELEMENTS: frozenset[str] = frozenset(
    {"O", "H", "C", "N", "S", "P", "F", "Cl", "Br", "I", "Se", "Te", "B"}
)

# Commodity precursor classes that a *simple* compound may belong to and still be
# considered buildable stock. (Class labels come from datasets.classify_precursor
# and are heuristic; the single-cation gate below does the real work.)
DEFAULT_ALLOWED_CLASSES: frozenset[str] = frozenset(
    {
        "oxide",
        "carbonate",
        "nitrate",
        "hydroxide",
        "acetate",
        "sulfate",
        "halide",
        "sulfide",
        "elemental_or_other",
    }
)

DEFAULT_MIN_FREQUENCY = 20
STOCK_CACHE_FILENAME = "stock.json"


def cation_elements(formula: str) -> set[str]:
    """Return the set of (metal) cation elements in ``formula``.

    A cation is any element that is not in :data:`ANION_ELEMENTS`. Robust to
    parse failures and hydrate dots via ``safe_element_set``.
    """
    return {el for el in safe_element_set(formula) if el not in ANION_ELEMENTS}


def is_simple_compound(formula: str) -> bool:
    """True if ``formula`` is a single-cation commodity (or an elemental phase).

    Binary oxides (MgO), single-cation carbonates/nitrates/hydroxides/acetates/
    sulfates/halides/sulfides, and elemental metals/non-metals all pass.
    Multi-cation compounds (BaTiO3, ZnFe2O4, LiFePO4, ...) fail and are therefore
    treated as recursion sub-targets rather than stock.
    """
    return len(cation_elements(formula)) <= 1


@dataclass(frozen=True)
class Stock:
    """A mined set of buildable precursors + the membership / solved criterion."""

    formulas: frozenset[str] = frozenset()
    allowed_classes: frozenset[str] = DEFAULT_ALLOWED_CLASSES
    min_frequency: int = DEFAULT_MIN_FREQUENCY

    def contains(self, precursor: PrecursorRecord | str) -> bool:
        """Membership test: is this precursor a buildable commodity (in stock)?

        A precursor is in stock iff it is a *simple* compound (single cation) AND
        either (a) its exact formula is in the mined high-frequency set, or (b) it
        belongs to an allowed commodity class. Multi-cation complex compounds are
        never in stock so they can become recursion sub-targets.
        """
        if isinstance(precursor, PrecursorRecord):
            formula = precursor.formula
            class_name = precursor.class_name
        else:
            formula = precursor
            class_name = None
        if not formula:
            return False
        if not is_simple_compound(formula):
            return False
        if formula in self.formulas:
            return True
        if class_name is not None:
            return class_name in self.allowed_classes
        # No class label available (bare formula): admit simple compounds since
        # a single-cation phase is a commodity by construction here.
        return True

    def size(self) -> int:
        return len(self.formulas)

    def to_dict(self) -> dict:
        return {
            "formulas": sorted(self.formulas),
            "allowed_classes": sorted(self.allowed_classes),
            "min_frequency": self.min_frequency,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Stock":
        return cls(
            formulas=frozenset(payload.get("formulas", [])),
            allowed_classes=frozenset(payload.get("allowed_classes", DEFAULT_ALLOWED_CLASSES)),
            min_frequency=int(payload.get("min_frequency", DEFAULT_MIN_FREQUENCY)),
        )


def precursor_frequency(routes: Iterable[RouteRecord]) -> Counter:
    """Count precursor-formula occurrences across a route corpus."""
    freq: Counter = Counter()
    for route in routes:
        for precursor in route.precursors:
            freq[precursor.formula] += 1
    return freq


def build_stock(
    routes: Iterable[RouteRecord],
    min_frequency: int = DEFAULT_MIN_FREQUENCY,
    allowed_classes: frozenset[str] = DEFAULT_ALLOWED_CLASSES,
) -> Stock:
    """Mine a :class:`Stock` from a route corpus.

    The explicit formula set is every precursor formula with frequency
    ``>= min_frequency`` that is also a *simple* single-cation commodity, so that
    frequent-but-complex phases (BaTiO3, ...) are excluded and stay recursion
    candidates. Pass a train-only corpus to obtain a leakage-controlled,
    target-disjoint stock for benchmarking (Phase 3).
    """
    freq = precursor_frequency(routes)
    formulas = frozenset(
        formula
        for formula, count in freq.items()
        if count >= min_frequency and is_simple_compound(formula)
    )
    return Stock(formulas=formulas, allowed_classes=allowed_classes, min_frequency=min_frequency)


def load_or_build_stock(
    processed_dir: str | Path,
    routes: Iterable[RouteRecord] | None = None,
    min_frequency: int = DEFAULT_MIN_FREQUENCY,
    rebuild: bool = False,
) -> Stock:
    """Load a cached stock from ``processed_dir/stock.json`` or build and cache it.

    When ``routes`` is None and no cache exists, both processed corpora are
    loaded to mine the stock. Set ``rebuild=True`` to force re-mining. The cache
    is keyed only by path, so callers that need a per-split (train-only) stock
    should call :func:`build_stock` directly rather than caching here.
    """
    processed_dir = Path(processed_dir)
    cache_path = processed_dir / STOCK_CACHE_FILENAME
    if cache_path.exists() and not rebuild:
        cached = Stock.from_dict(json.loads(cache_path.read_text()))
        if cached.min_frequency == min_frequency:
            return cached

    if routes is None:
        from .datasets import load_processed_routes

        routes = list(load_processed_routes(processed_dir, "solid_state"))
        try:
            routes += list(load_processed_routes(processed_dir, "hydrothermal"))
            routes += list(load_processed_routes(processed_dir, "precipitation"))
        except (FileNotFoundError, KeyError):
            pass

    stock = build_stock(routes, min_frequency=min_frequency)
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(stock.to_dict(), indent=2))
    return stock


def route_is_solved(precursors: Iterable[PrecursorRecord], stock: Stock) -> bool:
    """A single-step route is solved iff every leaf precursor is in stock."""
    precursors = list(precursors)
    if not precursors:
        return False
    return all(stock.contains(precursor) for precursor in precursors)


def stock_coverage(precursors: Iterable[PrecursorRecord], stock: Stock) -> float:
    """Fraction of leaf precursors that are in stock (partial-solve credit)."""
    precursors = list(precursors)
    if not precursors:
        return 0.0
    return sum(1 for p in precursors if stock.contains(p)) / len(precursors)
