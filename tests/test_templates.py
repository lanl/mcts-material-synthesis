"""Tests for template mining (WS-B)."""

import json

from synthesis_planner.core.schema import NumericRange, OperationRecord, PrecursorRecord, RouteRecord
from synthesis_planner.data.templates import (
    TemplateLibrary,
    family_key,
    load_or_build_templates,
    mine_templates,
    target_cation_count,
)


def _route(target, precursor_specs, temp=900.0, atmosphere="air", verbs=("mix", "heat")):
    precursors = tuple(PrecursorRecord(f, cls, tuple(c for c in f if c.isupper())) for f, cls in precursor_specs)
    ops = []
    for v in verbs:
        if v == "heat":
            ops.append(OperationRecord("heat", temperature_c=NumericRange(temp, temp, "C"),
                                       time_h=NumericRange(8.0, 8.0, "h"), atmosphere=atmosphere))
        else:
            ops.append(OperationRecord(v))
    return RouteRecord(
        route_id=f"r-{target}-{temp}", source_doi="10.1/x", publication_year=2020,
        modality="solid_state", target_formula=target,
        target_elements=tuple(c for c in target if c.isupper()), chemical_system="x",
        target_class="oxide", precursors=precursors, solvents=(), operations=tuple(ops),
        reaction_string="", paragraph_excerpt="", source_dataset="test",
    )


def test_target_cation_count_and_family_key():
    assert target_cation_count("BaTiO3") == 2
    assert target_cation_count("TiO2") == 1
    assert family_key("oxide", 2) == "oxide:2"


def test_mine_decomposition_patterns_ranked():
    routes = [_route("BaTiO3", [("BaCO3", "carbonate"), ("TiO2", "oxide")]) for _ in range(10)]
    routes += [_route("SrTiO3", [("SrO", "oxide"), ("TiO2", "oxide")]) for _ in range(3)]
    lib = mine_templates(routes)
    patterns = lib.precursor_class_patterns("oxide", 2)
    assert patterns
    # The carbonate+oxide pattern is most frequent -> highest prior.
    top = patterns[0]
    assert top.precursor_class_pattern == ("carbonate", "oxide")
    assert top.support == 10
    assert 0.0 < top.prior <= 1.0


def test_condition_schedule_percentiles_and_atmosphere():
    routes = [_route("BaTiO3", [("BaCO3", "carbonate"), ("TiO2", "oxide")], temp=t)
              for t in (800.0, 900.0, 1000.0, 1100.0, 1200.0)]
    lib = mine_templates(routes)
    sched = lib.condition_schedule("oxide", 2)
    assert sched is not None
    assert sched.temperature_p25 <= sched.temperature_p50 <= sched.temperature_p75
    assert sched.top_atmosphere == "air"
    assert sched.heating_steps_median == 1


def test_operation_sequences_compressed():
    routes = [_route("BaTiO3", [("BaCO3", "carbonate"), ("TiO2", "oxide")],
                     verbs=("mix", "mix", "heat")) for _ in range(5)]
    lib = mine_templates(routes)
    seqs = lib.operation_sequence_priors("oxide", 2)
    assert seqs
    # consecutive duplicate 'mix' collapsed
    assert seqs[0][0] == ("mix", "heat")


def test_fallback_lookup():
    lib = mine_templates([_route("BaTiO3", [("BaCO3", "carbonate"), ("TiO2", "oxide")])])
    # An unseen cation_count still resolves via the target_class / global fallback.
    assert lib.precursor_class_patterns("oxide", 5)
    assert lib.condition_schedule("nonexistent_class", 9) is not None


def test_roundtrip_serialization():
    lib = mine_templates([_route("BaTiO3", [("BaCO3", "carbonate"), ("TiO2", "oxide")])])
    restored = TemplateLibrary.from_dict(json.loads(json.dumps(lib.to_dict())))
    assert restored.precursor_class_patterns("oxide", 2)[0].precursor_class_pattern == ("carbonate", "oxide")


def test_load_or_build_caches(tmp_path):
    lib = mine_templates([_route("BaTiO3", [("BaCO3", "carbonate"), ("TiO2", "oxide")])])
    (tmp_path / "templates_solid_state.json").write_text(json.dumps(lib.to_dict()))
    loaded = load_or_build_templates(tmp_path, "solid_state")
    assert loaded.precursor_class_patterns("oxide", 2)
