"""DiffSynDiscoverySearch: MCTS outer loop over the choices DiffSyn conditions on.

The contribution (Strategy B, see DISCOVERY_DIFFSYN_INTEGRATION.md): DiffSyn maps
(framework, OSDA) -> a *distribution* of realistic continuous conditions, but it
does not choose the OSDA or the mineralizer route, score feasibility, or seek
novelty. This PUCT search supplies exactly those: it searches the discrete
decisions, calls the (frozen) generator for conditions via **progressive
widening** over its samples, scores each candidate recipe multi-objectively
(feasibility x novelty x realism), and returns a **diverse portfolio**.

Decision tree (shallow by design in Phase 1; OSDA *design* deepens it in Phase 3):

    root (target framework fixed)
      +-- OSDA choice        (children: one per candidate OSDA)
            +-- mineralizer   (children: OH route / F route)
                  +-- conditions   (progressive widening: each new child is one
                                    generator sample -> a terminal ZeoliteRecipe)

Each MCTS iteration descends root->OSDA->mineralizer, then either widens (draws a
fresh generator sample, evaluates its reward) or re-selects an existing sampled
recipe, and backs the reward up the path. No torch here -- the generator/oracle
are injected Protocols, so this whole file is tested against stubs.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .condition_generator import ConditionGenerator
from .novelty import ZeoliteNoveltyIndex
from .oracle import FeasibilityOracle
from .zeolite_schema import (
    ConditionSpec,
    DEFAULT_CONDITION_SPEC,
    MineralizerRoute,
    OSDA,
    ZeoliteRecipe,
    ZeoliteTarget,
)

DEFAULT_MINERALIZERS: tuple[MineralizerRoute, ...] = (MineralizerRoute("OH"), MineralizerRoute("F"))


@dataclass(frozen=True)
class SearchConfig:
    iterations: int = 200
    exploration_constant: float = 1.4
    # Progressive widening at a mineralizer node: allowed condition children after
    # v visits = min(pw_cap, ceil(pw_c * (v+1)^pw_alpha)).
    pw_cap: int = 16
    pw_c: float = 2.0
    pw_alpha: float = 0.5
    # Multi-objective reward weights (normalized internally). feasibility gates
    # novelty so "novel + infeasible" cannot win; realism keeps conditions
    # on-manifold. See the design doc's reward section.
    w_feasibility: float = 0.5
    w_novelty: float = 0.3
    w_realism: float = 0.2
    top_k: int = 5
    # Portfolio diversity: minimum normalized distance between two selected recipes.
    diversity_radius: float = 0.15
    seed: Optional[int] = 0


@dataclass
class RewardBreakdown:
    feasibility: float
    novelty: float
    realism: float
    total: float


class _Node:
    __slots__ = (
        "kind", "osda", "mineralizer", "recipe", "reward",
        "prior", "children", "visit_count", "total_value", "expanded", "pw_visits",
    )

    def __init__(self, kind, prior=1.0, osda=None, mineralizer=None, recipe=None, reward=None):
        self.kind = kind  # "root" | "osda" | "mineralizer" | "condition"
        self.osda: Optional[OSDA] = osda
        self.mineralizer: Optional[MineralizerRoute] = mineralizer
        self.recipe: Optional[ZeoliteRecipe] = recipe
        self.reward: Optional[RewardBreakdown] = reward
        self.prior = prior
        self.children: list[_Node] = []
        self.visit_count = 0
        self.total_value = 0.0
        self.expanded = False
        self.pw_visits = 0  # mineralizer-node visits, drives progressive widening

    @property
    def q_value(self) -> float:
        return self.total_value / self.visit_count if self.visit_count else 0.0


class DiffSynDiscoverySearch:
    """PUCT search coupling MCTS (outer, discrete) with DiffSyn (inner, conditions).

    Injected dependencies (all Protocols -> testable on stubs):
      * ``generator``: ConditionGenerator (DiffSyn or stub) -- samples conditions.
      * ``oracle``: FeasibilityOracle -- does (OSDA, conditions) template target?
      * ``novelty_index``: ZeoliteNoveltyIndex -- distance from the known corpus.
      * ``osda_pool``: candidate OSDAs the search chooses among (Phase 3: generate).
    """

    def __init__(
        self,
        generator: ConditionGenerator,
        oracle: FeasibilityOracle,
        novelty_index: ZeoliteNoveltyIndex,
        osda_pool: Sequence[OSDA],
        config: Optional[SearchConfig] = None,
        mineralizers: Sequence[MineralizerRoute] = DEFAULT_MINERALIZERS,
        spec: ConditionSpec = DEFAULT_CONDITION_SPEC,
        reward=None,
    ):
        if not osda_pool:
            raise ValueError("osda_pool must be non-empty")
        self.generator = generator
        self.oracle = oracle
        self.novelty_index = novelty_index
        self.osda_pool = list(osda_pool)
        self.config = config or SearchConfig()
        self.mineralizers = list(mineralizers)
        self.spec = spec
        # Optional pluggable multi-objective reward (discovery.reward.DiscoveryReward:
        # stability + DiffSyn feasibility + property + novelty). When None, the
        # search uses its built-in feasibility x novelty x realism reward. Any
        # object exposing ``.score(target, recipe, generator) -> obj-with-.total``
        # qualifies (duck-typed) so the two reward shapes are interchangeable.
        self.reward = reward
        self.rng = random.Random(self.config.seed)
        self._evaluated: list[_Node] = []  # all terminal condition nodes (for portfolio)

    # -- reward ------------------------------------------------------------
    def _reward(self, target: ZeoliteTarget, recipe: ZeoliteRecipe):
        if self.reward is not None:
            return self.reward.score(target, recipe, self.generator)
        c = self.config
        feas = self.oracle.score(target, recipe.osda, recipe.conditions)
        nov = self.novelty_index.novelty(recipe)
        real = self.generator.realism(target, recipe.osda, recipe.conditions)
        wsum = c.w_feasibility + c.w_novelty + c.w_realism
        total = (c.w_feasibility * feas + c.w_novelty * nov + c.w_realism * real) / wsum
        return RewardBreakdown(feasibility=feas, novelty=nov, realism=real, total=total)

    # -- tree ops ----------------------------------------------------------
    def _pw_limit(self, visits: int) -> int:
        c = self.config
        return min(c.pw_cap, math.ceil(c.pw_c * (visits + 1) ** c.pw_alpha))

    def _puct_select(self, node: _Node) -> _Node:
        c = self.config.exploration_constant
        parent_n = max(1, node.visit_count)
        best, best_score = None, float("-inf")
        for child in node.children:
            u = c * child.prior * math.sqrt(parent_n) / (1 + child.visit_count)
            score = child.q_value + u
            if score > best_score:
                best, best_score = child, score
        return best  # type: ignore[return-value]

    def _expand_osda(self, root: _Node) -> None:
        prior = 1.0 / len(self.osda_pool)
        root.children = [_Node("osda", prior=prior, osda=o) for o in self.osda_pool]
        root.expanded = True

    def _expand_mineralizer(self, node: _Node) -> None:
        prior = 1.0 / len(self.mineralizers)
        node.children = [
            _Node("mineralizer", prior=prior, osda=node.osda, mineralizer=m)
            for m in self.mineralizers
        ]
        node.expanded = True

    def _widen_condition(self, target: ZeoliteTarget, node: _Node) -> _Node:
        """Draw one fresh generator sample at a mineralizer node -> new terminal."""
        seed = None if self.config.seed is None else self.rng.randint(0, 2**31 - 1)
        cond = self.generator.sample(target, node.osda, 1, seed=seed)[0]
        recipe = ZeoliteRecipe(
            target=target, osda=node.osda, mineralizer=node.mineralizer, conditions=cond
        )
        reward = self._reward(target, recipe)
        child = _Node("condition", prior=1.0, osda=node.osda,
                      mineralizer=node.mineralizer, recipe=recipe, reward=reward)
        node.children.append(child)
        self._evaluated.append(child)
        return child

    # -- main loop ---------------------------------------------------------
    def run(self, target: ZeoliteTarget) -> list[ZeoliteRecipe]:
        """Run the search and return a diverse top-k portfolio of recipes."""
        self._evaluated = []
        root = _Node("root")
        for _ in range(self.config.iterations):
            self._iterate(target, root)
        return self._portfolio()

    def _iterate(self, target: ZeoliteTarget, root: _Node) -> None:
        path: list[_Node] = [root]

        # 1) root -> OSDA
        if not root.expanded:
            self._expand_osda(root)
        osda_node = self._puct_select(root)
        path.append(osda_node)

        # 2) OSDA -> mineralizer
        if not osda_node.expanded:
            self._expand_mineralizer(osda_node)
        min_node = self._puct_select(osda_node)
        path.append(min_node)

        # 3) mineralizer -> conditions (progressive widening)
        min_node.pw_visits += 1
        if len(min_node.children) < self._pw_limit(min_node.pw_visits):
            leaf = self._widen_condition(target, min_node)
        else:
            leaf = self._puct_select(min_node)
        path.append(leaf)

        # 4) backup the terminal reward
        value = leaf.reward.total if leaf.reward else 0.0
        for node in path:
            node.visit_count += 1
            node.total_value += value

    # -- portfolio ---------------------------------------------------------
    def _condition_distance(self, a: ZeoliteRecipe, b: ZeoliteRecipe) -> float:
        total, count = 0.0, 0
        for name in self.spec.variables:
            lo, hi = self.spec.ranges.get(name, (0.0, 1.0))
            span = (hi - lo) or 1.0
            total += abs(a.conditions.get(name) - b.conditions.get(name)) / span
            count += 1
        d = total / count if count else 0.0
        if a.osda.smiles != b.osda.smiles:
            d = min(1.0, d + 0.5)  # different template => clearly distinct
        return d

    def _portfolio(self) -> list[ZeoliteRecipe]:
        """Greedy max-reward selection with a diversity radius (MMR-style)."""
        ranked = sorted(self._evaluated, key=lambda n: n.reward.total, reverse=True)
        picked: list[_Node] = []
        for node in ranked:
            if len(picked) >= self.config.top_k:
                break
            if all(
                self._condition_distance(node.recipe, p.recipe) >= self.config.diversity_radius
                for p in picked
            ):
                picked.append(node)
        # Backfill if diversity was too strict to reach top_k.
        if len(picked) < self.config.top_k:
            for node in ranked:
                if node not in picked:
                    picked.append(node)
                if len(picked) >= self.config.top_k:
                    break
        return [n.recipe for n in picked]

    def reward_of(self, recipe: ZeoliteRecipe, target: ZeoliteTarget) -> RewardBreakdown:
        """Public helper: score a recipe with the same multi-objective reward."""
        return self._reward(target, recipe)
