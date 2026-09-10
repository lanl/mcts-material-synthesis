"""
Route-planning engine and immutable domain model.

This is the synthesis domain's *core*: the bespoke sync PUCT engine
(`MonteCarloTreeSearch`/`TreeNode`), the frozen dataclasses that describe a
planning problem and its evaluated routes (`schema`), the modality-aware action
grammar (`grammar`), the chemistry-aware scoring pipeline (`scoring`), hard
constraints (`constraints`), the retrieval-grounded judge (`judge`), and the
chemistry/formula primitives. These modules depend only on one another, so the
package forms a self-contained core with no dependency on the data or
orchestration layers.

© 2026. Triad National Security, LLC. All rights reserved.
"""

from .mcts import MonteCarloTreeSearch, TreeNode
from .schema import (
    Action,
    EvaluationConfig,
    LabConstraints,
    PlannedRoute,
    PlanningProblem,
    PlanningState,
    RouteRecord,
)

__all__ = [
    "MonteCarloTreeSearch",
    "TreeNode",
    "Action",
    "EvaluationConfig",
    "LabConstraints",
    "PlannedRoute",
    "PlanningProblem",
    "PlanningState",
    "RouteRecord",
]
