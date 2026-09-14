"""Tests for the modular multi-objective discovery reward (stability + DiffSyn
feasibility + pore-size property + novelty), and its use inside the search.

Offline against the stub generator. Locks: OSDA-binding-primary/e_hull-guardrail
stability, DiffSyn-distribution feasibility, pore-size property closeness, the
weighted composition, and that DiscoveryReward is a drop-in reward for the search.
"""

from synthesis_planner.discovery import (
    DiffSynFeasibilityScorer,
    DiffSynDiscoverySearch,
    DiscoveryReward,
    OSDA,
    PoreSizePropertyOracle,
    SearchConfig,
    StubConditionGenerator,
    ZeoliteNoveltyIndex,
    ZeoliteRecipe,
    ZeoliteStabilityOracle,
    ZeoliteTarget,
)
from synthesis_planner.discovery.zeolite_schema import MineralizerRoute


def _cond(gen, t, o):
    return gen.sample(t, o, 1, seed=0)[0]


# --- (a) stability: binding primary, e_hull guardrail -------------------------


def test_stability_neutral_when_no_signals():
    oracle = ZeoliteStabilityOracle()  # no binding fn, no hull provider
    t, o = ZeoliteTarget("CHA"), OSDA("CX")
    cond = _cond(StubConditionGenerator(), t, o)
    assert oracle.stability(t, o, cond) == 0.5


def test_stability_driven_by_binding_energy():
    # Strong (negative) binding for the good OSDA, weak for the bad one.
    def be(fw, smiles):
        return -20.0 if smiles == "GOOD" else +5.0

    oracle = ZeoliteStabilityOracle(binding_energy_fn=be)
    t = ZeoliteTarget("CHA")
    gen = StubConditionGenerator()
    good = oracle.stability(t, OSDA("GOOD"), _cond(gen, t, OSDA("GOOD")))
    bad = oracle.stability(t, OSDA("BAD"), _cond(gen, t, OSDA("BAD")))
    assert good > 0.5 > bad


def test_hull_is_only_a_guardrail_weight():
    # With binding neutral, a bad hull energy moves stability only modestly
    # (default guardrail weight 0.2), never dominating.
    class Hull:
        def energy_above_hull(self, framework_code):
            return 1.0  # very unstable -> favorability ~0

    oracle = ZeoliteStabilityOracle(hull_provider=Hull(), w_binding=0.8, w_hull=0.2)
    t, o = ZeoliteTarget("CHA"), OSDA("CX")
    cond = _cond(StubConditionGenerator(), t, o)
    s = oracle.stability(t, o, cond)
    # binding neutral 0.5 (weight .8) + hull ~0 (weight .2) => ~0.4, not near 0.
    assert 0.35 <= s <= 0.45


# --- (b) DiffSyn feasibility from the generated distribution -------------------


def test_feasibility_in_bounds_and_tightness():
    t, o = ZeoliteTarget("CHA"), OSDA("CX")
    tight = DiffSynFeasibilityScorer(n_samples=64).feasibility(t, o, StubConditionGenerator(spread=0.05))
    loose = DiffSynFeasibilityScorer(n_samples=64).feasibility(t, o, StubConditionGenerator(spread=0.4))
    assert 0.0 < loose < tight <= 1.0  # tighter distribution => more feasible


def test_feasibility_is_cached_per_pair():
    scorer = DiffSynFeasibilityScorer(n_samples=16)
    t, o = ZeoliteTarget("MFI"), OSDA("CY")
    gen = StubConditionGenerator()
    a = scorer.feasibility(t, o, gen)
    b = scorer.feasibility(t, o, gen)
    assert a == b and (t.framework_code, o.smiles) in scorer._cache


def test_feasibility_penalizes_out_of_bounds_generator():
    # A generator that emits out-of-range values -> in_bounds < 1 -> lower feasibility.
    from synthesis_planner.discovery.zeolite_schema import ConditionVector, DEFAULT_CONDITION_SPEC

    class OOBGen:
        spec = DEFAULT_CONDITION_SPEC

        def sample(self, target, osda, n, *, seed=None):
            bad = {name: 1e6 for name in self.spec.variables}  # far out of range
            return [ConditionVector(bad, self.spec) for _ in range(n)]

        def realism(self, *a, **k):
            return 0.0

    f = DiffSynFeasibilityScorer(n_samples=8).feasibility(ZeoliteTarget("CHA"), OSDA("CX"), OOBGen())
    assert f == 0.0  # nothing in bounds


# --- (c) pore-size property ---------------------------------------------------


def test_pore_size_property_closeness():
    pore = {"CHA": 3.8, "FAU": 7.4}
    oracle = PoreSizePropertyOracle(pore_size_fn=lambda fw: pore.get(fw), target_range=(3.0, 4.0), scale=2.0)
    gen = StubConditionGenerator()
    in_range = oracle.property_match(ZeoliteTarget("CHA"), _cond(gen, ZeoliteTarget("CHA"), OSDA("X")))
    out_range = oracle.property_match(ZeoliteTarget("FAU"), _cond(gen, ZeoliteTarget("FAU"), OSDA("X")))
    assert in_range == 1.0
    assert 0.0 <= out_range < 1.0  # 7.4 is 3.4 above hi=4.0 over scale 2 -> 0


def test_pore_size_neutral_without_fn():
    oracle = PoreSizePropertyOracle(pore_size_fn=None, target_range=(3.0, 4.0))
    gen = StubConditionGenerator()
    assert oracle.property_match(ZeoliteTarget("CHA"), _cond(gen, ZeoliteTarget("CHA"), OSDA("X"))) == 0.5


# --- composition + search integration -----------------------------------------


def _discovery_reward(binding_fn=None, pore=None, ref=None):
    return DiscoveryReward(
        stability_oracle=ZeoliteStabilityOracle(binding_energy_fn=binding_fn),
        feasibility_scorer=DiffSynFeasibilityScorer(n_samples=32),
        property_oracle=PoreSizePropertyOracle(
            pore_size_fn=(lambda fw: (pore or {}).get(fw)), target_range=(3.0, 4.0)
        ),
        novelty_index=ZeoliteNoveltyIndex(reference=ref or {}),
    )


def test_discovery_reward_breakdown_and_bounds():
    gen = StubConditionGenerator()
    t, o = ZeoliteTarget("CHA"), OSDA("CX")
    rb = _discovery_reward(pore={"CHA": 3.5}).score(t, ZeoliteRecipe(t, o, MineralizerRoute("OH"), _cond(gen, t, o)), gen)
    for v in (rb.stability, rb.feasibility, rb.property, rb.novelty, rb.total):
        assert 0.0 <= v <= 1.0
    assert rb.property == 1.0  # 3.5 is inside (3.0, 4.0)


def test_discovery_reward_plugs_into_search():
    gen = StubConditionGenerator()

    def be(fw, smiles):
        return -15.0 if smiles == "CB" else 0.0

    reward = _discovery_reward(binding_fn=be, pore={"CHA": 3.5})
    search = DiffSynDiscoverySearch(
        generator=gen,
        oracle=None,  # unused when reward is provided
        novelty_index=ZeoliteNoveltyIndex(reference={}),
        osda_pool=[OSDA("CA"), OSDA("CB"), OSDA("CC")],
        config=SearchConfig(iterations=300, top_k=1, seed=0),
        reward=reward,
    )
    portfolio = search.run(ZeoliteTarget("CHA"))
    # The binding-favoured OSDA should surface as the top recipe.
    assert portfolio[0].osda.smiles == "CB"
    rb = search.reward_of(portfolio[0], ZeoliteTarget("CHA"))
    assert hasattr(rb, "total") and 0.0 <= rb.total <= 1.0
