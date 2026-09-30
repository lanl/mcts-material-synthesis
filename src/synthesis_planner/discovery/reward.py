"""Modular multi-objective discovery reward for zeolite synthesis discovery.

Composes four pluggable terms into one reward the MCTS search optimizes:

    R = w_s * Stability  +  w_f * SynthesisFeasibility  +  w_p * PropertyMatch  +  w_n * Novelty

Each term is a small object with a documented seam, so a domain swaps providers
without touching the search:

  (a) Stability  -- ``ZeoliteStabilityOracle``: **OSDA-framework binding energy is
      the primary "will it form" signal** (the DiffSyn paper's DFT signal), with
      MACE/MP energy-above-hull as a **low-weight guardrail**. This is the honest
      choice for zeolites, which are metastable microporous frameworks by design,
      so e_above_hull is only weakly discriminative among them (it is the *strong*
      signal for the dense-oxide pipeline, where the same class is reused with an
      e_hull-dominant weighting). Any unavailable sub-signal stays neutral (0.5),
      so a missing MP/MACE hookup never penalizes -- same discipline as the Run F
      physics provider.

  (b) SynthesisFeasibility  -- ``DiffSynFeasibilityScorer``: DiffSyn emits no
      scalar, so feasibility is read off the *quality of its generated condition
      distribution* for a (framework, OSDA) pair: physicality (fraction of samples
      in valid ranges) x tightness (a confident, low-spread distribution implies a
      well-determined recipe exists). This makes DiffSyn a *scorer*, not only a
      generator.

  (c) PropertyMatch  -- ``PoreSizePropertyOracle``: closeness of the framework's
      largest-included-sphere (pore) diameter to a target range -- the canonical
      zeolite application property, read from the 143-dim Zeo++ descriptors.

  (d) Novelty  -- the existing ``ZeoliteNoveltyIndex`` (distance from ZeoSyn),
      which flips the objective from recall to discovery. Gated by (a)-(c) so
      "novel but infeasible/unstable/off-spec" cannot win.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, runtime_checkable

from ..core.physics import stability_favorability
from .condition_generator import ConditionGenerator
from .novelty import ZeoliteNoveltyIndex
from .oracle import binding_energy_favorability
from .zeolite_schema import (
    ConditionSpec,
    ConditionVector,
    DEFAULT_CONDITION_SPEC,
    OSDA,
    ZeoliteRecipe,
    ZeoliteTarget,
)


# --- (a) stability ------------------------------------------------------------


@runtime_checkable
class HullStabilityProvider(Protocol):
    """Energy-above-hull provider for a zeolite framework (eV/atom, 0 = on hull).

    Real implementation: relax/energy the framework CIF with MACE, compare to the
    MP convex hull for its composition (or a cached hull). Returns None when
    unavailable so the guardrail stays neutral.
    """

    def energy_above_hull(self, framework_code: str) -> Optional[float]:
        ...


class ZeoliteStabilityOracle:
    """Stability term: OSDA-binding primary + e_above_hull guardrail (per user choice).

    ``binding_energy_fn(framework_code, osda_smiles) -> kJ/mol/Si | None`` is the
    primary signal (DFT or a learned surrogate). ``hull_provider`` is the optional
    weak guardrail. Weights default to 0.8 / 0.2. Missing signals are neutral 0.5.
    """

    def __init__(
        self,
        binding_energy_fn: Optional[Callable[[str, str], Optional[float]]] = None,
        hull_provider: Optional[HullStabilityProvider] = None,
        w_binding: float = 0.8,
        w_hull: float = 0.2,
    ):
        self._be_fn = binding_energy_fn
        self._hull = hull_provider
        self.w_binding = w_binding
        self.w_hull = w_hull

    def _binding_favorability(self, target: ZeoliteTarget, osda: OSDA) -> float:
        if self._be_fn is None:
            return 0.5
        try:
            be = self._be_fn(target.framework_code, osda.smiles)
        except Exception:
            be = None
        return binding_energy_favorability(be)

    def _hull_favorability(self, target: ZeoliteTarget) -> float:
        if self._hull is None:
            return 0.5
        try:
            ehull = self._hull.energy_above_hull(target.framework_code)
        except Exception:
            ehull = None
        return stability_favorability(ehull, available=ehull is not None)

    def stability(self, target: ZeoliteTarget, osda: OSDA, conditions: ConditionVector) -> float:
        b = self._binding_favorability(target, osda)
        h = self._hull_favorability(target)
        wsum = self.w_binding + self.w_hull
        return (self.w_binding * b + self.w_hull * h) / wsum


# --- (b) synthesis feasibility from DiffSyn -----------------------------------


class DiffSynFeasibilityScorer:
    """Feasibility from the *quality* of DiffSyn's generated condition distribution.

    For a (framework, OSDA) pair: draw ``n_samples`` conditions, then
    ``feasibility = in_bounds * tightness`` where
      * ``in_bounds`` = fraction of samples with all 12 variables inside their
        physical ranges (physicality), and
      * ``tightness`` = 1 / (1 + mean range-normalized std) in (0, 1] (a confident,
        low-spread distribution implies a well-determined, realizable recipe).
    Cached per pair (the distribution is a property of the pair, not a single
    recipe), so it is computed once and reused across that pair's condition leaves.
    """

    def __init__(self, n_samples: int = 64, spec: ConditionSpec = DEFAULT_CONDITION_SPEC, seed: int = 0):
        self.n_samples = n_samples
        self.spec = spec
        self.seed = seed
        self._cache: dict[tuple[str, str], float] = {}

    def feasibility(self, target: ZeoliteTarget, osda: OSDA, generator: ConditionGenerator) -> float:
        key = (target.framework_code, osda.smiles)
        if key in self._cache:
            return self._cache[key]
        samples = generator.sample(target, osda, self.n_samples, seed=self.seed)
        cols = {name: [s.get(name) for s in samples] for name in self.spec.variables}

        in_bounds_count = 0
        for s in samples:
            ok = True
            for name in self.spec.variables:
                lo, hi = self.spec.ranges.get(name, (float("-inf"), float("inf")))
                if not (lo <= s.get(name) <= hi):
                    ok = False
                    break
            in_bounds_count += int(ok)
        in_bounds = in_bounds_count / len(samples) if samples else 0.0

        norm_stds = []
        for name in self.spec.variables:
            lo, hi = self.spec.ranges.get(name, (0.0, 1.0))
            span = (hi - lo) or 1.0
            col = cols[name]
            std = statistics.pstdev(col) if len(col) > 1 else 0.0
            norm_stds.append(std / span)
        mean_norm_std = sum(norm_stds) / len(norm_stds) if norm_stds else 0.0
        tightness = 1.0 / (1.0 + mean_norm_std)

        value = in_bounds * tightness
        self._cache[key] = value
        return value


# --- (c) target property ------------------------------------------------------


def _range_closeness(value: float, lo: float, hi: float, scale: float) -> float:
    """1.0 inside [lo, hi]; graded decay outside over ``scale`` units."""
    if lo <= value <= hi:
        return 1.0
    dist = (lo - value) if value < lo else (value - hi)
    return max(0.0, 1.0 - dist / scale)


class PoreSizePropertyOracle:
    """Property term: closeness of the framework's pore diameter to a target range.

    ``pore_size_fn(framework_code) -> Angstrom | None`` reads the largest-included-
    sphere descriptor (``zeo_largest_included_sphere`` in the 143-dim Zeo++ vector).
    ``target_range`` is the desired pore diameter window (Angstrom). Neutral 0.5
    when the pore size is unavailable, so a missing descriptor never penalizes.
    """

    def __init__(
        self,
        pore_size_fn: Optional[Callable[[str], Optional[float]]],
        target_range: tuple[float, float],
        scale: float = 3.0,
    ):
        self._fn = pore_size_fn
        self.target_range = target_range
        self.scale = scale

    def property_match(self, target: ZeoliteTarget, conditions: ConditionVector) -> float:
        if self._fn is None:
            return 0.5
        try:
            pore = self._fn(target.framework_code)
        except Exception:
            pore = None
        if pore is None:
            return 0.5
        lo, hi = self.target_range
        return _range_closeness(float(pore), lo, hi, self.scale)


# --- composed reward ----------------------------------------------------------


@dataclass
class DiscoveryRewardBreakdown:
    stability: float
    feasibility: float
    property: float
    novelty: float
    total: float


class DiscoveryReward:
    """Weighted composition of the four discovery terms; the object the search scores.

    Duck-types the search's reward interface (returns an object with ``.total``),
    so ``DiffSynDiscoverySearch(reward=DiscoveryReward(...))`` swaps it in for the
    built-in feasibility x novelty x realism reward with no other change.
    """

    def __init__(
        self,
        stability_oracle: ZeoliteStabilityOracle,
        feasibility_scorer: DiffSynFeasibilityScorer,
        property_oracle: PoreSizePropertyOracle,
        novelty_index: ZeoliteNoveltyIndex,
        w_stability: float = 0.35,
        w_feasibility: float = 0.30,
        w_property: float = 0.15,
        w_novelty: float = 0.20,
    ):
        self.stability_oracle = stability_oracle
        self.feasibility_scorer = feasibility_scorer
        self.property_oracle = property_oracle
        self.novelty_index = novelty_index
        self.w_stability = w_stability
        self.w_feasibility = w_feasibility
        self.w_property = w_property
        self.w_novelty = w_novelty

    def score(
        self, target: ZeoliteTarget, recipe: ZeoliteRecipe, generator: ConditionGenerator
    ) -> DiscoveryRewardBreakdown:
        s = self.stability_oracle.stability(target, recipe.osda, recipe.conditions)
        f = self.feasibility_scorer.feasibility(target, recipe.osda, generator)
        p = self.property_oracle.property_match(target, recipe.conditions)
        n = self.novelty_index.novelty(recipe)
        wsum = self.w_stability + self.w_feasibility + self.w_property + self.w_novelty
        total = (
            self.w_stability * s + self.w_feasibility * f + self.w_property * p + self.w_novelty * n
        ) / wsum
        return DiscoveryRewardBreakdown(stability=s, feasibility=f, property=p, novelty=n, total=total)
