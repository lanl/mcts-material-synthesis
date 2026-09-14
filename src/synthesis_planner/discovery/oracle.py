"""Feasibility oracle seam: does (OSDA, conditions) plausibly template the target?

DiffSyn deliberately outputs no feasibility/synthesizability score -- the paper
grounds recipes post hoc with **DFT OSDA-framework binding energies** and
experiment. So the discovery search needs a separate oracle to keep novelty
honest (novel + feasible, not novel + hallucinated).

- ``FeasibilityOracle`` -- the Protocol the search scores against.
- ``BindingEnergyOracle`` -- the real path: maps an OSDA-framework binding energy
  (kJ/mol per Si, more negative = better templating, per the paper) to [0,1].
  The ``binding_energy_fn`` seam is where a DFT calc or a learned surrogate plugs
  in; unset -> the oracle stays neutral (0.5) rather than guessing.
- ``CooccurrenceOracle`` -- an offline stub/prior: an OSDA that co-occurs with the
  framework in ZeoSyn is more feasible than one that never does. Cheap, data-only,
  good enough to drive/test the search; not a physics oracle.
"""

from __future__ import annotations

import math
from typing import Callable, Mapping, Optional, Protocol, runtime_checkable

from .zeolite_schema import ConditionVector, OSDA, ZeoliteTarget


@runtime_checkable
class FeasibilityOracle(Protocol):
    def score(self, target: ZeoliteTarget, osda: OSDA, conditions: ConditionVector) -> float:
        """[0,1] feasibility of this recipe templating the target framework."""
        ...

    def binding_energy(self, target: ZeoliteTarget, osda: OSDA) -> Optional[float]:
        """OSDA-framework binding energy (kJ/mol/Si) if available, else None."""
        ...


# Binding energy (kJ/mol per Si) scale for the logistic map. ~ -10 is a strong
# templating interaction; 0 is indifferent. Tunable; documented, not load-bearing.
_BE_SCALE_KJ = 10.0


def binding_energy_favorability(be_kj_per_si: Optional[float]) -> float:
    """Map binding energy to [0,1] (more negative = better). Neutral 0.5 if None."""
    if be_kj_per_si is None:
        return 0.5
    # logistic in (-BE/scale): -10 -> ~0.73, 0 -> 0.5, +10 -> ~0.27
    return 1.0 / (1.0 + math.exp(be_kj_per_si / _BE_SCALE_KJ))


class BindingEnergyOracle:
    """Real feasibility oracle backed by an OSDA-framework binding-energy function.

    ``binding_energy_fn(framework_code, osda_smiles) -> kJ/mol/Si | None`` is the
    plug point for a DFT calculation (expensive; e.g. the paper's protocol) or a
    learned surrogate (search-scale). When it is None or returns None, the oracle
    is neutral (0.5) so an unknown never fabricates feasibility.
    """

    def __init__(self, binding_energy_fn: Optional[Callable[[str, str], Optional[float]]] = None):
        self._fn = binding_energy_fn

    def binding_energy(self, target: ZeoliteTarget, osda: OSDA) -> Optional[float]:
        if self._fn is None:
            return None
        try:
            return self._fn(target.framework_code, osda.smiles)
        except Exception:
            return None

    def score(self, target: ZeoliteTarget, osda: OSDA, conditions: ConditionVector) -> float:
        return binding_energy_favorability(self.binding_energy(target, osda))


class CooccurrenceOracle:
    """Offline prior: OSDA-framework empirical co-occurrence in ZeoSyn.

    ``cooccurrence[framework_code]`` is a mapping ``osda_smiles -> weight`` (e.g.
    normalized count of literature recipes pairing them). Feasibility = the
    normalized weight; unseen pairs get ``floor`` (a small non-zero prior so the
    search can still explore genuinely novel pairings). This is a data-only
    stand-in for a physics oracle -- honest as a prior, not as ground truth.
    """

    def __init__(self, cooccurrence: Mapping[str, Mapping[str, float]], floor: float = 0.05):
        self._co = cooccurrence
        self.floor = floor

    def binding_energy(self, target: ZeoliteTarget, osda: OSDA) -> Optional[float]:
        return None  # not a physics oracle

    def score(self, target: ZeoliteTarget, osda: OSDA, conditions: ConditionVector) -> float:
        table = self._co.get(target.framework_code, {})
        if not table:
            return self.floor
        weight = float(table.get(osda.smiles, 0.0))
        top = max(table.values()) if table else 1.0
        norm = weight / top if top > 0 else 0.0
        return max(self.floor, norm)
