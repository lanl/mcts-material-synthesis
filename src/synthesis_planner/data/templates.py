"""Backward decomposition-template + condition-schedule mining (WS-B).

Organic retrosynthesis applies *reaction templates backward*. The inorganic
analog, mined here, is a distribution over **precursor-class multiset patterns**
conditioned on the target family (target class + cation count), together with
per-family **condition schedules** (temperature percentiles, dwell, atmosphere,
regrind count) and **operation-sequence priors**. These widen the MCTS action
space so search has something to explore beyond a single retrieval set.

Everything is cheap deterministic aggregation over the processed JSONL. Precursor
class labels come from ``datasets.classify_precursor`` and are heuristic; mining
tolerates that noise by ranking patterns by empirical support and always keeping
coarse fallbacks (by target class, then global).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Iterable

from ..core.schema import RouteRecord
from .stock import cation_elements


def family_key(target_class: str, cation_count: int) -> str:
    """A coarse family key: e.g. 'oxide:2' for a ternary (2-cation) oxide."""
    return f"{target_class}:{cation_count}"


def target_cation_count(formula: str) -> int:
    return len(cation_elements(formula))


@dataclass(frozen=True)
class DecompositionTemplate:
    """A mined target-family -> precursor-class-multiset pattern with support."""

    precursor_class_pattern: tuple[str, ...]  # sorted multiset of precursor classes
    support: int
    prior: float  # support normalized within the family (0-1)


@dataclass(frozen=True)
class ConditionSchedule:
    """Per-family heating condition distribution (solid-state / anneal steps)."""

    temperature_p25: float | None
    temperature_p50: float | None
    temperature_p75: float | None
    dwell_h_median: float | None
    top_atmosphere: str | None
    heating_steps_median: int
    support: int


@dataclass
class TemplateLibrary:
    """Mined templates + condition/operation priors, with graded fallbacks."""

    decomposition_templates: dict[str, list[DecompositionTemplate]] = field(default_factory=dict)
    condition_schedules: dict[str, ConditionSchedule] = field(default_factory=dict)
    operation_sequences: dict[str, list[tuple[tuple[str, ...], int]]] = field(default_factory=dict)

    # ---- lookups (with fallback: family -> target_class -> global) ----------

    def precursor_class_patterns(self, target_class: str, cation_count: int) -> list[DecompositionTemplate]:
        for key in (family_key(target_class, cation_count), target_class, "*"):
            if key in self.decomposition_templates:
                return self.decomposition_templates[key]
        return []

    def condition_schedule(self, target_class: str, cation_count: int) -> ConditionSchedule | None:
        for key in (family_key(target_class, cation_count), target_class, "*"):
            if key in self.condition_schedules:
                return self.condition_schedules[key]
        return None

    def operation_sequence_priors(self, target_class: str, cation_count: int) -> list[tuple[tuple[str, ...], int]]:
        for key in (family_key(target_class, cation_count), target_class, "*"):
            if key in self.operation_sequences:
                return self.operation_sequences[key]
        return []

    # ---- serialization ------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "decomposition_templates": {
                key: [asdict(tpl) for tpl in tpls]
                for key, tpls in self.decomposition_templates.items()
            },
            "condition_schedules": {key: asdict(sch) for key, sch in self.condition_schedules.items()},
            "operation_sequences": {
                key: [[list(seq), count] for seq, count in seqs]
                for key, seqs in self.operation_sequences.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TemplateLibrary":
        decomposition = {
            key: [DecompositionTemplate(tuple(t["precursor_class_pattern"]), t["support"], t["prior"]) for t in tpls]
            for key, tpls in payload.get("decomposition_templates", {}).items()
        }
        schedules = {
            key: ConditionSchedule(**sch) for key, sch in payload.get("condition_schedules", {}).items()
        }
        sequences = {
            key: [(tuple(seq), count) for seq, count in seqs]
            for key, seqs in payload.get("operation_sequences", {}).items()
        }
        return cls(decomposition, schedules, sequences)


def _percentile(sorted_values: list[float], p: float) -> float | None:
    if not sorted_values:
        return None
    idx = min(len(sorted_values) - 1, max(0, int(round(p * (len(sorted_values) - 1)))))
    return round(sorted_values[idx], 1)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return round(ordered[mid], 1)
    return round((ordered[mid - 1] + ordered[mid]) / 2.0, 1)


def mine_templates(
    routes: Iterable[RouteRecord],
    max_patterns_per_family: int = 8,
    max_sequences_per_family: int = 6,
) -> TemplateLibrary:
    """Aggregate decomposition patterns + condition/operation priors from a corpus.

    Pass a train-only corpus for a leakage-controlled template set (Phase 3).
    """
    pattern_counts: dict[str, Counter] = defaultdict(Counter)
    temps: dict[str, list[float]] = defaultdict(list)
    dwells: dict[str, list[float]] = defaultdict(list)
    atmospheres: dict[str, Counter] = defaultdict(Counter)
    heating_step_counts: dict[str, list[int]] = defaultdict(list)
    seq_counts: dict[str, Counter] = defaultdict(Counter)

    def _keys(target_class: str, cation_count: int) -> tuple[str, ...]:
        # Emit every granularity so fallbacks are always populated.
        return (family_key(target_class, cation_count), target_class, "*")

    for route in routes:
        cation_count = target_cation_count(route.target_formula)
        pattern = tuple(sorted(p.class_name for p in route.precursors))
        keys = _keys(route.target_class, cation_count)

        for key in keys:
            if pattern:
                pattern_counts[key][pattern] += 1

        heating_ops = [op for op in route.operations if op.verb == "heat"]
        for key in keys:
            heating_step_counts[key].append(len(heating_ops))
        for op in heating_ops:
            if op.temperature_c and op.temperature_c.midpoint is not None:
                for key in keys:
                    temps[key].append(op.temperature_c.midpoint)
            if op.time_h and op.time_h.midpoint is not None:
                for key in keys:
                    dwells[key].append(op.time_h.midpoint)
            if op.atmosphere:
                for key in keys:
                    atmospheres[key][op.atmosphere] += 1

        verbs = tuple(op.verb for op in route.operations)
        # Compress consecutive duplicate verbs to capture the operation *shape*.
        compressed: list[str] = []
        for v in verbs:
            if not compressed or compressed[-1] != v:
                compressed.append(v)
        if compressed:
            for key in keys:
                seq_counts[key][tuple(compressed)] += 1

    decomposition_templates: dict[str, list[DecompositionTemplate]] = {}
    for key, counter in pattern_counts.items():
        total = sum(counter.values())
        ranked = counter.most_common(max_patterns_per_family)
        decomposition_templates[key] = [
            DecompositionTemplate(pattern, support, round(support / total, 4) if total else 0.0)
            for pattern, support in ranked
        ]

    condition_schedules: dict[str, ConditionSchedule] = {}
    families = set(temps) | set(dwells) | set(heating_step_counts) | set(atmospheres)
    for key in families:
        ordered_temps = sorted(temps.get(key, []))
        steps = heating_step_counts.get(key, [])
        atmo = atmospheres.get(key)
        condition_schedules[key] = ConditionSchedule(
            temperature_p25=_percentile(ordered_temps, 0.25),
            temperature_p50=_percentile(ordered_temps, 0.50),
            temperature_p75=_percentile(ordered_temps, 0.75),
            dwell_h_median=_median(dwells.get(key, [])),
            top_atmosphere=atmo.most_common(1)[0][0] if atmo else None,
            heating_steps_median=int(_median([float(s) for s in steps]) or 0) if steps else 0,
            support=len(steps),
        )

    operation_sequences: dict[str, list[tuple[tuple[str, ...], int]]] = {}
    for key, counter in seq_counts.items():
        operation_sequences[key] = [(seq, count) for seq, count in counter.most_common(max_sequences_per_family)]

    return TemplateLibrary(decomposition_templates, condition_schedules, operation_sequences)


TEMPLATE_CACHE_FILENAME = "templates.json"


def load_or_build_templates(
    processed_dir: str | Path,
    modality: str = "solid_state",
    routes: Iterable[RouteRecord] | None = None,
    rebuild: bool = False,
) -> TemplateLibrary:
    """Load cached templates from ``processed_dir/templates_<modality>.json`` or mine + cache."""
    processed_dir = Path(processed_dir)
    cache_path = processed_dir / f"templates_{modality}.json"
    if cache_path.exists() and not rebuild:
        return TemplateLibrary.from_dict(json.loads(cache_path.read_text()))

    if routes is None:
        from .datasets import load_processed_routes

        routes = list(load_processed_routes(processed_dir, modality))

    library = mine_templates(routes)
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(library.to_dict(), indent=2))
    return library
