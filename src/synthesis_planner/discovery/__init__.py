"""Generative zeolite synthesis-discovery (Strategy B: MCTS x DiffSyn).

MCTS searches the discrete synthesis decisions (OSDA, mineralizer route) that the
frozen, pretrained DiffSyn diffusion model conditions on but does not itself
choose; DiffSyn supplies realistic continuous conditions; a feasibility oracle
grounds novelty. See ``DISCOVERY_DIFFSYN_INTEGRATION.md`` for the full design and
handoff notes.

Nothing here imports torch at module load. The real DiffSyn model is reached only
through ``DiffSynConditionGenerator`` (lazy import); all other code is tested
against ``StubConditionGenerator`` / stub oracles offline.
"""

from __future__ import annotations

from .condition_generator import (
    ConditionGenerator,
    DiffSynConditionGenerator,
    StubConditionGenerator,
)
from .novelty import ZeoliteNoveltyIndex, build_reference_from_recipes
from .oracle import (
    BindingEnergyOracle,
    CooccurrenceOracle,
    FeasibilityOracle,
    binding_energy_favorability,
)
from .reward import (
    DiffSynFeasibilityScorer,
    DiscoveryReward,
    DiscoveryRewardBreakdown,
    HullStabilityProvider,
    PoreSizePropertyOracle,
    ZeoliteStabilityOracle,
)
from .search import DiffSynDiscoverySearch, RewardBreakdown, SearchConfig
from .zeolite_schema import (
    CONDITION_VARIABLES,
    ConditionSpec,
    ConditionVector,
    DEFAULT_CONDITION_SPEC,
    MineralizerRoute,
    OSDA,
    ZeoliteRecipe,
    ZeoliteTarget,
)

__all__ = [
    "ConditionGenerator",
    "StubConditionGenerator",
    "DiffSynConditionGenerator",
    "FeasibilityOracle",
    "BindingEnergyOracle",
    "CooccurrenceOracle",
    "binding_energy_favorability",
    "ZeoliteNoveltyIndex",
    "build_reference_from_recipes",
    "DiffSynDiscoverySearch",
    "SearchConfig",
    "RewardBreakdown",
    "DiscoveryReward",
    "DiscoveryRewardBreakdown",
    "ZeoliteStabilityOracle",
    "DiffSynFeasibilityScorer",
    "PoreSizePropertyOracle",
    "HullStabilityProvider",
    "ZeoliteTarget",
    "OSDA",
    "MineralizerRoute",
    "ConditionVector",
    "ConditionSpec",
    "DEFAULT_CONDITION_SPEC",
    "CONDITION_VARIABLES",
    "ZeoliteRecipe",
]
