"""Core dataclasses shared across the planner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class NumericRange:
    minimum: float | None
    maximum: float | None
    units: str | None = None

    @property
    def midpoint(self) -> float | None:
        if self.minimum is None and self.maximum is None:
            return None
        if self.minimum is None:
            return self.maximum
        if self.maximum is None:
            return self.minimum
        return (self.minimum + self.maximum) / 2.0


@dataclass(frozen=True)
class PrecursorRecord:
    formula: str
    class_name: str
    elements: tuple[str, ...]


@dataclass(frozen=True)
class OperationRecord:
    verb: str
    temperature_c: NumericRange | None = None
    time_h: NumericRange | None = None
    atmosphere: str | None = None
    source_label: str | None = None


@dataclass(frozen=True)
class RouteRecord:
    route_id: str
    source_doi: str
    publication_year: int | None
    modality: str
    target_formula: str
    target_elements: tuple[str, ...]
    chemical_system: str
    target_class: str
    precursors: tuple[PrecursorRecord, ...]
    solvents: tuple[str, ...]
    operations: tuple[OperationRecord, ...]
    reaction_string: str
    paragraph_excerpt: str
    source_dataset: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LabConstraints:
    min_temperature_c: float | None = None
    max_temperature_c: float | None = None
    allowed_atmospheres: tuple[str, ...] = field(default_factory=tuple)
    forbidden_precursor_classes: tuple[str, ...] = field(default_factory=tuple)
    max_precursors: int = 6
    max_heating_steps: int = 3
    require_mixing: bool = True


@dataclass(frozen=True)
class PlanningProblem:
    target_formula: str
    modality: str = "solid_state"
    max_precursors: int = 6
    lab_constraints: LabConstraints = field(default_factory=LabConstraints)


@dataclass(frozen=True)
class JudgeResult:
    score: float
    notes: tuple[str, ...]
    flags: tuple[str, ...]
    evidence_dois: tuple[str, ...] = field(default_factory=tuple)
    rubric_scores: dict[str, float] = field(default_factory=dict)
    uncertainty: float = 0.0


@dataclass(frozen=True)
class BalancedSpecies:
    formula: str
    coefficient: float


@dataclass(frozen=True)
class ReactionBalanceResult:
    feasible: bool
    framework_match_fraction: float
    precursor_coefficients: tuple[float, ...]
    environmental_reactants: tuple[BalancedSpecies, ...] = field(default_factory=tuple)
    byproducts: tuple[BalancedSpecies, ...] = field(default_factory=tuple)
    unused_precursors: tuple[str, ...] = field(default_factory=tuple)
    residual_elements: dict[str, float] = field(default_factory=dict)
    equation: str | None = None


@dataclass(frozen=True)
class RedoxAnalysisResult:
    target_charge: float | None
    precursor_charge: float | None
    required_direction: str
    environment_support: str
    notes: tuple[str, ...] = field(default_factory=tuple)
    flags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ThermoAnalysisResult:
    score: float
    gas_release_moles: float
    gas_uptake_moles: float
    byproduct_count: int
    decomposition_match: float
    redox_match: float
    notes: tuple[str, ...] = field(default_factory=tuple)
    hull_energy_ev_per_atom: float | None = None
    formation_energy_ev_per_atom: float | None = None
    decomposition_energy_ev_per_atom: float | None = None
    competing_phases: tuple[str, ...] = field(default_factory=tuple)
    is_stable: bool | None = None
    reaction_driving_force_ev: float | None = None
    is_exothermic: bool | None = None


@dataclass(frozen=True)
class HardCheckResult:
    valid: bool
    flags: tuple[str, ...]
    notes: tuple[str, ...]
    coverage_fraction: float
    blocking_flags: tuple[str, ...]
    reaction_balance: ReactionBalanceResult | None = None
    redox: RedoxAnalysisResult | None = None


@dataclass(frozen=True)
class ScoreBreakdown:
    validity: float
    stoich: float
    precursor: float
    thermo: float
    retrieval: float
    condition: float
    llm: float
    cost: float
    hazard: float
    complexity: float
    total: float
    # Solved-to-stock term: fraction of leaf precursors in stock (retrosynthesis
    # redesign). Dominant terminal reward; 0.0 when no stock context is supplied.
    stock: float = 0.0


@dataclass(frozen=True)
class EvaluationConfig:
    judge_name: str = "deterministic"
    use_judge: bool = True
    use_hard_checks: bool = True
    use_partial_judge: bool = False
    judge_config: dict[str, Any] = field(default_factory=dict)
    # Solved-to-stock context and reward weights (retrosynthesis redesign).
    # ``stock`` is duck-typed (a data.stock.Stock) to avoid a core->data import
    # cycle; it must expose ``.contains(precursor)``. When present, the scorer
    # rewards routes whose precursors bottom out in stock and demotes the
    # self-similarity ``retrieval`` term so it guides expansion, not the leaf.
    stock: Any = None
    stock_weight: float = 2.5
    retrieval_weight: float = 0.2


@dataclass(frozen=True)
class PlannedRoute:
    target_formula: str
    modality: str
    precursors: tuple[PrecursorRecord, ...]
    solvents: tuple[str, ...]
    operations: tuple[OperationRecord, ...]
    evidence_dois: tuple[str, ...]
    analog_targets: tuple[str, ...]
    hard_checks: HardCheckResult
    score: ScoreBreakdown
    thermo: ThermoAnalysisResult
    judge: JudgeResult
    mcts_value: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Action:
    kind: str
    label: str
    prior: float
    payload: Any


@dataclass(frozen=True)
class PlanningState:
    problem: PlanningProblem
    target_elements: tuple[str, ...]
    target_class: str
    stage: str = "precursors"
    precursors: tuple[PrecursorRecord, ...] = field(default_factory=tuple)
    solvents: tuple[str, ...] = field(default_factory=tuple)
    operations: tuple[OperationRecord, ...] = field(default_factory=tuple)
    evidence_dois: tuple[str, ...] = field(default_factory=tuple)
    analog_targets: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_terminal(self) -> bool:
        return self.stage == "terminal"


@dataclass
class DAGNode:
    """A node in a synthesis DAG: a material that must be made.

    Mutable (unlike the frozen planning dataclasses) because the DAG is assembled
    incrementally by the recursion driver. ``recipe`` is the chosen single-step
    route for this material; ``children`` are the sub-targets (non-stock
    precursors) expanded further. A node is a solved leaf when ``in_stock`` is
    True; ``dangling`` marks a non-stock precursor that could not be expanded
    (depth cap hit or no plausible sub-route).
    """

    target_formula: str
    modality: str
    depth: int
    in_stock: bool = False
    dangling: bool = False
    recipe: "PlannedRoute | None" = None
    children: list["DAGNode"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def leaves(self) -> list["DAGNode"]:
        """All leaf nodes reachable from this node (self if it is a leaf)."""
        if self.is_leaf:
            return [self]
        collected: list[DAGNode] = []
        for child in self.children:
            collected.extend(child.leaves())
        return collected

    def node_count(self) -> int:
        return 1 + sum(child.node_count() for child in self.children)

    def max_depth(self) -> int:
        if not self.children:
            return self.depth
        return max(child.max_depth() for child in self.children)


@dataclass
class SynthesisDAG:
    """A full synthesis plan for a root target, as a recursive DAG of DAGNodes."""

    target_formula: str
    modality: str
    root: DAGNode
    max_depth_cap: int = 2

    @property
    def is_solved(self) -> bool:
        """Solved iff every leaf is in stock and every internal node has a recipe."""
        leaves = self.root.leaves()
        if not leaves:
            return False
        if not all(leaf.in_stock and not leaf.dangling for leaf in leaves):
            return False
        return all(self._node_valid(node) for node in self._internal_nodes())

    def _node_valid(self, node: DAGNode) -> bool:
        if node.recipe is None:
            return False
        return node.recipe.hard_checks.valid

    def _internal_nodes(self) -> list[DAGNode]:
        return [node for node in self._all_nodes() if node.children]

    def _all_nodes(self) -> list[DAGNode]:
        stack = [self.root]
        collected: list[DAGNode] = []
        while stack:
            node = stack.pop()
            collected.append(node)
            stack.extend(node.children)
        return collected

    @property
    def depth(self) -> int:
        return self.root.max_depth()

    @property
    def node_count(self) -> int:
        return self.root.node_count()

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_formula": self.target_formula,
            "modality": self.modality,
            "is_solved": self.is_solved,
            "depth": self.depth,
            "node_count": self.node_count,
            "root": _dag_node_to_dict(self.root),
        }


def _dag_node_to_dict(node: DAGNode) -> dict[str, Any]:
    return {
        "target_formula": node.target_formula,
        "modality": node.modality,
        "depth": node.depth,
        "in_stock": node.in_stock,
        "dangling": node.dangling,
        "precursors": [p.formula for p in node.recipe.precursors] if node.recipe else [],
        "valid": node.recipe.hard_checks.valid if node.recipe else None,
        "children": [_dag_node_to_dict(child) for child in node.children],
    }
