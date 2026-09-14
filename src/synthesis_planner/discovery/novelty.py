"""Novelty index: distance of a proposed recipe from the known corpus (ZeoSyn).

This is the term that flips the objective from *recall* to *discovery*. A recipe
that reproduces a literature recipe scores ~0 novelty; one that is far from every
known recipe (for the same framework) scores ~1. Novelty is only ever rewarded in
the search when multiplied/added against feasibility + realism, so "far from the
corpus" cannot win on its own -- that is the whole point (novelty gated by
physics, not novelty for its own sake).

The corpus is injected as a list of reference recipes per framework (built from
ZeoSyn in Phase 2; a small in-memory set suffices for tests). Distance is over the
normalized 12-dim condition vector plus a discrete OSDA-identity term.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

from .zeolite_schema import ConditionSpec, ConditionVector, DEFAULT_CONDITION_SPEC, ZeoliteRecipe


class ZeoliteNoveltyIndex:
    """Nearest-neighbour novelty over a reference corpus of known recipes.

    ``reference`` maps ``framework_code -> list[(osda_smiles, ConditionVector)]``
    of known literature recipes. ``ranges`` (from the spec) normalize condition
    distances so no single high-magnitude variable (e.g. Si/Al) dominates.
    """

    def __init__(
        self,
        reference: Mapping[str, list[tuple[str, ConditionVector]]],
        spec: ConditionSpec = DEFAULT_CONDITION_SPEC,
        osda_mismatch_bonus: float = 0.25,
    ):
        self._ref = reference
        self.spec = spec
        self.osda_mismatch_bonus = osda_mismatch_bonus

    def _norm_condition_distance(self, a: ConditionVector, b: ConditionVector) -> float:
        """Mean absolute per-variable difference, range-normalized to ~[0,1]."""
        total = 0.0
        count = 0
        for name in self.spec.variables:
            lo, hi = self.spec.ranges.get(name, (0.0, 1.0))
            span = (hi - lo) or 1.0
            total += abs(a.get(name) - b.get(name)) / span
            count += 1
        return total / count if count else 0.0

    def novelty(self, recipe: ZeoliteRecipe) -> float:
        """[0,1] novelty = distance to the nearest known recipe for this framework.

        No references for the framework at all -> fully novel (1.0). Same OSDA as a
        neighbour compares conditions directly; a different OSDA adds a discrete
        mismatch bonus (a new template is itself a form of novelty).
        """
        refs = self._ref.get(recipe.target.framework_code)
        if not refs:
            return 1.0
        best = 1.0
        for osda_smiles, cond in refs:
            d = self._norm_condition_distance(recipe.conditions, cond)
            if osda_smiles != recipe.osda.smiles:
                d = min(1.0, d + self.osda_mismatch_bonus)
            best = min(best, d)
        return max(0.0, min(1.0, best))


def build_reference_from_recipes(
    recipes: Iterable[tuple[str, str, ConditionVector]],
) -> dict[str, list[tuple[str, ConditionVector]]]:
    """Helper: fold ``(framework_code, osda_smiles, ConditionVector)`` rows into the
    per-framework reference mapping ``ZeoliteNoveltyIndex`` expects.

    In Phase 2 this is populated from ZeoSyn; ``ConditionVector``s are the recorded
    gel ratios + conditions per literature recipe.
    """
    ref: dict[str, list[tuple[str, ConditionVector]]] = {}
    for framework_code, osda_smiles, cond in recipes:
        ref.setdefault(framework_code, []).append((osda_smiles, cond))
    return ref
