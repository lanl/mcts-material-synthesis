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


@dataclass(frozen=True)
class EvaluationConfig:
    judge_name: str = "deterministic"
    use_judge: bool = True
    use_hard_checks: bool = True
    judge_config: dict[str, Any] = field(default_factory=dict)


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
