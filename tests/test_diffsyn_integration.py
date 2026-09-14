"""Tests for the MCTS x DiffSyn zeolite discovery integration (Strategy B).

All offline: the heavy DiffSyn model is behind the ConditionGenerator seam and is
exercised here via StubConditionGenerator, which is faithful to DiffSyn's exact
12-variable output contract. These tests lock the coupling, reward, progressive
widening, novelty-vs-recall behaviour, and portfolio diversity so the real weights
are a drop-in later (see DISCOVERY_DIFFSYN_INTEGRATION.md).
"""

import pytest

from synthesis_planner.discovery import (
    CONDITION_VARIABLES,
    ConditionVector,
    CooccurrenceOracle,
    DiffSynDiscoverySearch,
    OSDA,
    SearchConfig,
    StubConditionGenerator,
    ZeoliteNoveltyIndex,
    ZeoliteRecipe,
    ZeoliteTarget,
    binding_energy_favorability,
    build_reference_from_recipes,
)
from synthesis_planner.discovery.oracle import BindingEnergyOracle
from synthesis_planner.discovery.zeolite_schema import DEFAULT_CONDITION_SPEC, MineralizerRoute


def _osda(tag):
    return OSDA(smiles=f"C{tag}", name=f"osda-{tag}")


def _empty_novelty():
    return ZeoliteNoveltyIndex(reference={})


# --- schema / contract --------------------------------------------------------


def test_condition_spec_is_exact_diffsyn_12_vars():
    assert len(CONDITION_VARIABLES) == 12
    assert CONDITION_VARIABLES[:2] == ("Si/Al", "Al/P")
    assert CONDITION_VARIABLES[-2:] == ("cryst_temp", "cryst_time")


def test_stub_generator_matches_contract_and_is_deterministic():
    gen = StubConditionGenerator()
    t, o = ZeoliteTarget("CHA"), _osda("A")
    a = gen.sample(t, o, 5, seed=0)
    b = gen.sample(t, o, 5, seed=0)
    assert len(a) == 5
    assert set(a[0].values.keys()) == set(CONDITION_VARIABLES)
    assert [x.as_tuple() for x in a] == [x.as_tuple() for x in b]  # deterministic


def test_stub_generator_varies_by_pair():
    gen = StubConditionGenerator()
    ca = gen.sample(ZeoliteTarget("CHA"), _osda("A"), 1, seed=0)[0].as_tuple()
    cb = gen.sample(ZeoliteTarget("MFI"), _osda("B"), 1, seed=0)[0].as_tuple()
    assert ca != cb  # genuine pair-to-pair variation, like real DiffSyn


def test_stub_realism_peaks_at_cloud_center():
    gen = StubConditionGenerator()
    t, o = ZeoliteTarget("CHA"), _osda("A")
    center = {name: gen._center_and_std(t, o, name)[0] for name in CONDITION_VARIABLES}
    at_center = gen.realism(t, o, ConditionVector(center))
    samples = gen.sample(t, o, 20, seed=1)
    assert at_center >= max(gen.realism(t, o, s) for s in samples) - 1e-9
    assert 0.0 <= at_center <= 1.0


# --- oracle / novelty ---------------------------------------------------------


def test_binding_energy_favorability_monotonic():
    assert binding_energy_favorability(None) == 0.5
    assert binding_energy_favorability(-10.0) > binding_energy_favorability(0.0)
    assert binding_energy_favorability(0.0) > binding_energy_favorability(+10.0)


def test_binding_energy_oracle_neutral_without_fn():
    oracle = BindingEnergyOracle(binding_energy_fn=None)
    s = oracle.score(ZeoliteTarget("CHA"), _osda("A"),
                     StubConditionGenerator().sample(ZeoliteTarget("CHA"), _osda("A"), 1)[0])
    assert s == 0.5


def test_cooccurrence_oracle_prefers_known_pairs():
    oracle = CooccurrenceOracle({"CHA": {"CX": 10.0, "CY": 1.0}}, floor=0.05)
    cond = StubConditionGenerator().sample(ZeoliteTarget("CHA"), OSDA("CX"), 1)[0]
    known = oracle.score(ZeoliteTarget("CHA"), OSDA("CX"), cond)
    unknown = oracle.score(ZeoliteTarget("CHA"), OSDA("CZ"), cond)
    assert known == 1.0 and unknown == 0.05


def test_novelty_full_when_no_reference_and_low_when_matching():
    gen = StubConditionGenerator()
    t, o = ZeoliteTarget("CHA"), _osda("A")
    cond = gen.sample(t, o, 1, seed=0)[0]
    recipe = ZeoliteRecipe(t, o, MineralizerRoute("OH"), cond)
    assert _empty_novelty().novelty(recipe) == 1.0
    # An identical recipe present in the corpus -> ~0 novelty.
    ref = build_reference_from_recipes([("CHA", o.smiles, cond)])
    idx = ZeoliteNoveltyIndex(reference=ref)
    assert idx.novelty(recipe) < 1e-9


# --- search coupling ----------------------------------------------------------


def _search(oracle=None, novelty=None, config=None, pool=None):
    gen = StubConditionGenerator()
    pool = pool or [_osda("A"), _osda("B"), _osda("C")]
    return DiffSynDiscoverySearch(
        generator=gen,
        oracle=oracle or BindingEnergyOracle(None),
        novelty_index=novelty or _empty_novelty(),
        osda_pool=pool,
        config=config or SearchConfig(iterations=120, top_k=3, seed=0),
    )


def test_search_runs_and_returns_portfolio():
    search = _search()
    portfolio = search.run(ZeoliteTarget("CHA"))
    assert 1 <= len(portfolio) <= 3
    assert all(isinstance(r, ZeoliteRecipe) for r in portfolio)
    # Every recipe carries a full 12-var condition vector and a pool OSDA.
    assert all(set(r.conditions.values.keys()) == set(CONDITION_VARIABLES) for r in portfolio)
    assert all(r.osda.smiles in {"CA", "CB", "CC"} for r in portfolio)


def test_progressive_widening_bounds_condition_samples():
    # With few iterations, each mineralizer node holds at most pw_cap condition
    # children; the number of distinct sampled recipes is bounded, not 1-per-iter.
    cfg = SearchConfig(iterations=200, pw_cap=4, pw_c=1.0, pw_alpha=0.5, seed=0, top_k=5)
    search = _search(config=cfg)
    search.run(ZeoliteTarget("CHA"))
    # total evaluated leaves <= pool(3) * mineralizers(2) * pw_cap(4)
    assert len(search._evaluated) <= 3 * 2 * 4


def test_search_is_deterministic_under_seed():
    p1 = _search(config=SearchConfig(iterations=100, top_k=3, seed=7)).run(ZeoliteTarget("MFI"))
    p2 = _search(config=SearchConfig(iterations=100, top_k=3, seed=7)).run(ZeoliteTarget("MFI"))
    assert [r.key() for r in p1] == [r.key() for r in p2]


def test_feasibility_oracle_steers_osda_selection():
    # An oracle that strongly favours OSDA 'CB' should make the search visit it
    # most; check the top portfolio recipe uses the favoured template.
    oracle = CooccurrenceOracle({"CHA": {"CB": 10.0}}, floor=0.01)
    search = _search(oracle=oracle, config=SearchConfig(iterations=300, top_k=1, seed=0))
    portfolio = search.run(ZeoliteTarget("CHA"))
    assert portfolio[0].osda.smiles == "CB"


def test_novelty_reward_shifts_selection_away_from_corpus():
    """Discovery, not recall: with novelty weighted, a recipe matching the corpus
    should score below an off-corpus one for the same feasibility/realism."""
    gen = StubConditionGenerator()
    t = ZeoliteTarget("CHA")
    o = _osda("A")
    corpus_cond = gen.sample(t, o, 1, seed=0)[0]
    ref = build_reference_from_recipes([("CHA", o.smiles, corpus_cond)])
    search = DiffSynDiscoverySearch(
        generator=gen,
        oracle=BindingEnergyOracle(None),  # neutral 0.5 feasibility everywhere
        novelty_index=ZeoliteNoveltyIndex(reference=ref),
        osda_pool=[o],
        config=SearchConfig(iterations=10, w_feasibility=0.0, w_novelty=1.0, w_realism=0.0, seed=0),
    )
    on_corpus = search.reward_of(ZeoliteRecipe(t, o, MineralizerRoute("OH"), corpus_cond), t)
    far_cond = gen.sample(t, o, 40, seed=3)
    off_corpus_best = max(
        search.reward_of(ZeoliteRecipe(t, o, MineralizerRoute("OH"), c), t).total for c in far_cond
    )
    assert off_corpus_best > on_corpus.total


def test_empty_pool_rejected():
    with pytest.raises(ValueError):
        DiffSynDiscoverySearch(
            generator=StubConditionGenerator(),
            oracle=BindingEnergyOracle(None),
            novelty_index=_empty_novelty(),
            osda_pool=[],
        )


def test_diffsyn_generator_import_is_torch_free():
    # Importing the adapter must not require torch (lazy). Constructing is fine;
    # only .sample()/.realism() touch torch.
    from synthesis_planner.discovery import DiffSynConditionGenerator

    gen = DiffSynConditionGenerator(repo_root="/nonexistent")
    assert gen.device == "cpu" and gen._model is None
