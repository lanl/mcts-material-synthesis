"""Zeolite synthesis-discovery domain model (Strategy B: MCTS x DiffSyn).

Self-contained domain types for the zeolite discovery loop. Deliberately separate
from the oxide ``core`` schema: zeolite synthesis is parameterised completely
differently (framework topology + organic template + gel-composition ratios), and
mixing the two would pollute both. See ``DISCOVERY_DIFFSYN_INTEGRATION.md``.

The ``ConditionSpec`` below is the EXACT output contract of the released DiffSyn
checkpoint ``runs/diff/system/run1`` (verified from ``eltonpan/zeosyn_gen``
``data/syn_variables.py``): a 12-dim continuous vector = 10 gel ratios + 2
crystallisation conditions, in this order. Nothing here imports torch; the heavy
model lives behind the ``ConditionGenerator`` seam in ``condition_generator.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

# --- DiffSyn output variables (exact order = model column order) --------------
# 10 gel-composition ratios (syn_variables.ratios) then 2 conditions
# (syn_variables.conds). This ordering matches
# ``dataset.ratio_names + dataset.cond_names`` used to inverse-transform samples.
GEL_RATIO_NAMES: tuple[str, ...] = (
    "Si/Al", "Al/P", "Si/Ge", "Si/B", "Na/T", "K/T", "OH/T", "F/T", "H2O/T", "sda1/T",
)
CONDITION_NAMES: tuple[str, ...] = ("cryst_temp", "cryst_time")
CONDITION_VARIABLES: tuple[str, ...] = GEL_RATIO_NAMES + CONDITION_NAMES  # len 12


@dataclass(frozen=True)
class ConditionSpec:
    """Names (and optional physical ranges) of the generated condition variables.

    ``variables`` is the ordered tuple the generator emits. ``ranges`` gives
    optional (lo, hi) physical bounds per variable, used only for sanity-clipping
    and for the stub generator; the real DiffSyn output is already in physical
    units after inverse quantile-transform, so ranges are advisory.
    """

    variables: tuple[str, ...] = CONDITION_VARIABLES
    ranges: Mapping[str, tuple[float, float]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.variables)

    def index(self, name: str) -> int:
        return self.variables.index(name)


# Advisory physical ranges (only for the offline stub / clipping; not from the
# model). Rough zeolite-synthesis envelopes; refine against ZeoSyn in Phase 2.
DEFAULT_CONDITION_RANGES: dict[str, tuple[float, float]] = {
    "Si/Al": (1.0, 500.0), "Al/P": (0.0, 2.0), "Si/Ge": (0.0, 100.0), "Si/B": (0.0, 100.0),
    "Na/T": (0.0, 3.0), "K/T": (0.0, 2.0), "OH/T": (0.0, 2.0), "F/T": (0.0, 2.0),
    "H2O/T": (1.0, 100.0), "sda1/T": (0.0, 2.0),
    "cryst_temp": (25.0, 250.0), "cryst_time": (0.0, 60.0),  # temp degC, time days
}

DEFAULT_CONDITION_SPEC = ConditionSpec(variables=CONDITION_VARIABLES, ranges=DEFAULT_CONDITION_RANGES)


@dataclass(frozen=True)
class ConditionVector:
    """One concrete synthesis-condition point: variable name -> physical value.

    A single sample from the generator (one row of DiffSyn's 12-dim output). Kept
    as a mapping (not a bare tuple) so downstream code is order-independent and
    self-documenting.
    """

    values: Mapping[str, float]
    spec: ConditionSpec = DEFAULT_CONDITION_SPEC

    def as_tuple(self) -> tuple[float, ...]:
        return tuple(float(self.values[name]) for name in self.spec.variables)

    def get(self, name: str) -> float:
        return float(self.values[name])


@dataclass(frozen=True)
class ZeoliteTarget:
    """A target zeolite framework to discover a synthesis route for.

    ``framework_code`` is the 3-letter IZA code (e.g. "CHA", "MFI", "UFI").
    ``descriptors`` optionally carries the 143-dim Zeo++ feature vector DiffSyn
    conditions on (``syn_variables.zeo_cols`` order); when absent, the real
    generator looks it up by code from ``data/zeolite_descriptors.csv``.
    """

    framework_code: str
    descriptors: tuple[float, ...] = ()  # 143-dim zeo_cols vector, optional here


@dataclass(frozen=True)
class OSDA:
    """An organic structure-directing agent (template).

    ``smiles`` is the canonical identifier. ``descriptors`` optionally carries the
    14-dim RDKit descriptor vector DiffSyn conditions on
    (``syn_variables.osda_cols`` order); when absent, the real generator computes
    or looks it up. ``name`` is a human label (optional).
    """

    smiles: str
    descriptors: tuple[float, ...] = ()  # 14-dim osda_cols vector, optional here
    name: str = ""


@dataclass(frozen=True)
class MineralizerRoute:
    """Discrete mineralizer choice searched by MCTS (hydroxide vs fluoride route).

    Encoded as a light tag the search branches on; it biases which gel ratios
    matter (OH/T vs F/T) and is passed through to scoring/novelty, not to DiffSyn
    directly (DiffSyn already emits both OH/T and F/T).
    """

    tag: str  # "OH" or "F"


@dataclass(frozen=True)
class ZeoliteRecipe:
    """A fully-specified candidate synthesis route = target + choices + conditions.

    This is the terminal object the discovery search scores and returns in the
    portfolio: which framework, which OSDA, which mineralizer route, and one
    concrete condition vector sampled from the generator.
    """

    target: ZeoliteTarget
    osda: OSDA
    mineralizer: MineralizerRoute
    conditions: ConditionVector

    def key(self) -> tuple:
        """Hashable identity for dedup/diversity (rounded conditions)."""
        rounded = tuple(round(v, 3) for v in self.conditions.as_tuple())
        return (self.target.framework_code, self.osda.smiles, self.mineralizer.tag, rounded)
