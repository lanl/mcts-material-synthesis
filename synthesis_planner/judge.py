"""Pluggable retrieval-grounded route judges."""

from __future__ import annotations

import json
import os
from typing import Any

from .formula import safe_required_target_elements
from .schema import HardCheckResult, JudgeResult, PlanningState, RouteRecord


class BaseJudge:
    name = "base"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def evaluate(self, state: PlanningState, analogs: list[tuple[float, RouteRecord]], hard_checks: HardCheckResult) -> JudgeResult:
        raise NotImplementedError


class NullJudge(BaseJudge):
    name = "none"

    def evaluate(self, state: PlanningState, analogs: list[tuple[float, RouteRecord]], hard_checks: HardCheckResult) -> JudgeResult:
        return JudgeResult(score=0.0, notes=(), flags=(), evidence_dois=(), rubric_scores={}, uncertainty=1.0)


class DeterministicJudge(BaseJudge):
    name = "deterministic"

    def evaluate(self, state: PlanningState, analogs: list[tuple[float, RouteRecord]], hard_checks: HardCheckResult) -> JudgeResult:
        evidence_dois = tuple(route.source_doi for _, route in analogs[:3] if route.source_doi)
        notes = []
        flags = list(hard_checks.flags)
        rubric = {
            "precursor_plausibility": 0.5,
            "condition_compatibility": 0.5,
            "operation_completeness": 0.5,
            "literature_analogy": min(1.0, max((score for score, _ in analogs), default=0.0) / 8.0),
            "practicality": 0.6,
        }

        formulas = [precursor.formula for precursor in state.precursors]
        ops = [operation.verb for operation in state.operations]
        heating = [operation for operation in state.operations if operation.verb == "heat"]
        temperatures = [op.temperature_c.midpoint for op in heating if op.temperature_c and op.temperature_c.midpoint is not None]
        atmospheres = [op.atmosphere.lower() for op in heating if op.atmosphere]

        score = 0.5

        if state.problem.modality == "solid_state":
            if any("CO3" in formula or "NO3" in formula for formula in formulas):
                if temperatures and min(temperatures) >= 600.0:
                    notes.append("Salt precursors have a plausible decomposition window for solid-state synthesis.")
                    score += 0.1
                    rubric["precursor_plausibility"] += 0.1
                else:
                    flags.append("decomposition_risk")
                    notes.append("Low-temperature calcination may leave carbonates or nitrates incompletely decomposed.")
                    score -= 0.15

            if len(safe_required_target_elements(state.problem.target_formula)) >= 3 and "anneal" not in [operation.source_label for operation in state.operations if operation.source_label]:
                flags.append("limited_diffusion_support")
                notes.append("A multicomponent solid-state target may benefit from regrinding and a second anneal.")
                score -= 0.1
            else:
                rubric["operation_completeness"] += 0.15

            if "mix" in ops:
                rubric["operation_completeness"] += 0.1

        elif state.problem.modality == "hydrothermal":
            if state.solvents:
                notes.append("The route includes an explicit solution medium for hydrothermal processing.")
                rubric["operation_completeness"] += 0.1
            if temperatures and all(80.0 <= temp <= 280.0 for temp in temperatures):
                notes.append("Hydrothermal hold temperatures fall in a plausible autoclave range.")
                score += 0.1
                rubric["condition_compatibility"] += 0.15
            if "wash" in ops and "dry" in ops:
                rubric["operation_completeness"] += 0.15

        elif state.problem.modality == "precipitation":
            if state.solvents:
                notes.append("The route includes an explicit solution medium for precipitation chemistry.")
                rubric["operation_completeness"] += 0.1
            if "precipitate" in ops:
                rubric["operation_completeness"] += 0.15
            if "wash" in ops and "dry" in ops:
                rubric["operation_completeness"] += 0.15
            else:
                flags.append("incomplete_postprocessing")
                notes.append("Precipitation routes usually need wash and dry steps before termination.")
                score -= 0.1

        if not hard_checks.valid:
            score -= 0.15
            rubric["practicality"] -= 0.2
            notes.append("Hard validity checks indicate one or more blocking issues.")

        if any(atm in {"air", "oxygen", "o2"} for atm in atmospheres) and state.target_class in {"sulfide", "nitride"}:
            flags.append("atmosphere_mismatch")
            notes.append("Oxidizing atmosphere is suspicious for oxygen-sensitive target chemistry.")
            score -= 0.15

        score = max(0.0, min(score, 1.0))
        uncertainty = max(0.0, min(1.0, 0.8 - 0.1 * len(evidence_dois) + (0.2 if not hard_checks.valid else 0.0)))
        rubric = {key: max(0.0, min(value, 1.0)) for key, value in rubric.items()}
        if not notes:
            notes.append("The route is chemically plausible at a coarse heuristic level.")
        return JudgeResult(
            score=score,
            notes=tuple(notes),
            flags=tuple(dict.fromkeys(flags)),
            evidence_dois=evidence_dois,
            rubric_scores=rubric,
            uncertainty=uncertainty,
        )


class OpenAICompatibleStructuredJudge(BaseJudge):
    name = "openai_structured"
    _schema = {
        "type": "object",
        "properties": {
            "score": {"type": "number", "minimum": 0, "maximum": 1},
            "notes": {"type": "array", "items": {"type": "string"}},
            "flags": {"type": "array", "items": {"type": "string"}},
            "evidence_dois": {"type": "array", "items": {"type": "string"}},
            "rubric_scores": {
                "type": "object",
                "properties": {
                    "precursor_plausibility": {"type": "number", "minimum": 0, "maximum": 1},
                    "condition_compatibility": {"type": "number", "minimum": 0, "maximum": 1},
                    "operation_completeness": {"type": "number", "minimum": 0, "maximum": 1},
                    "literature_analogy": {"type": "number", "minimum": 0, "maximum": 1},
                    "practicality": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": [
                    "precursor_plausibility",
                    "condition_compatibility",
                    "operation_completeness",
                    "literature_analogy",
                    "practicality",
                ],
                "additionalProperties": False,
            },
            "uncertainty": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["score", "notes", "flags", "evidence_dois", "rubric_scores", "uncertainty"],
        "additionalProperties": False,
    }

    def evaluate(self, state: PlanningState, analogs: list[tuple[float, RouteRecord]], hard_checks: HardCheckResult) -> JudgeResult:
        payload = self._request_structured_judgment(state, analogs, hard_checks)
        return JudgeResult(
            score=_clamp(payload.get("score", 0.0)),
            notes=tuple(payload.get("notes", ())) or ("No model judge notes were returned.",),
            flags=tuple(dict.fromkeys(payload.get("flags", ()))),
            evidence_dois=tuple(payload.get("evidence_dois", ())),
            rubric_scores={key: _clamp(value) for key, value in payload.get("rubric_scores", {}).items()},
            uncertainty=_clamp(payload.get("uncertainty", 1.0)),
        )

    def _request_structured_judgment(
        self,
        state: PlanningState,
        analogs: list[tuple[float, RouteRecord]],
        hard_checks: HardCheckResult,
    ) -> dict[str, Any]:
        client = self._build_client()
        model = self.config.get("model") or "gpt-4o-mini"
        response = client.responses.create(
            model=model,
            instructions=self._instructions(),
            input=json.dumps(self._build_context_payload(state, analogs, hard_checks), indent=2),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "route_judgment",
                    "strict": True,
                    "schema": self._schema,
                }
            },
        )
        payload = json.loads(response.output_text)
        if not isinstance(payload, dict):
            raise ValueError("Structured judge response was not a JSON object.")
        return payload

    def _build_client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required for judge='openai_structured'. "
                "Install it with `pip install openai` or switch back to the deterministic judge."
            ) from exc

        api_key = self.config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No API key was found for judge='openai_structured'. "
                "Set `judge.api_key` in config.py or export OPENAI_API_KEY."
            )

        client_kwargs = {"api_key": api_key}
        if self.config.get("base_url"):
            client_kwargs["base_url"] = self.config["base_url"]
        return OpenAI(**client_kwargs)

    def _instructions(self) -> str:
        return (
            "You are an inorganic materials synthesis judge. "
            "Review the proposed route using the supplied target, route chemistry, hard-check analysis, "
            "and retrieved literature analogs. Score only the proposed route. "
            "Return valid JSON matching the schema exactly. "
            "Use evidence_dois only from the retrieved analog list. "
            "Keep notes brief and specific. "
            "Flag chemistry, feasibility, or literature-support problems when present."
        )

    def _build_context_payload(
        self,
        state: PlanningState,
        analogs: list[tuple[float, RouteRecord]],
        hard_checks: HardCheckResult,
    ) -> dict[str, Any]:
        analog_payload = []
        for score, route in analogs[:5]:
            analog_payload.append(
                {
                    "retrieval_score": round(score, 3),
                    "target_formula": route.target_formula,
                    "source_doi": route.source_doi,
                    "precursors": [precursor.formula for precursor in route.precursors],
                    "operations": [operation.source_label or operation.verb for operation in route.operations],
                    "reaction_string": route.reaction_string,
                    "paragraph_excerpt": route.paragraph_excerpt[:800],
                }
            )

        return {
            "target": {
                "formula": state.problem.target_formula,
                "modality": state.problem.modality,
                "target_class": state.target_class,
                "required_framework_elements": sorted(safe_required_target_elements(state.problem.target_formula)),
            },
            "candidate_route": {
                "precursors": [{"formula": precursor.formula, "class_name": precursor.class_name} for precursor in state.precursors],
                "solvents": list(state.solvents),
                "operations": [
                    {
                        "verb": operation.verb,
                        "source_label": operation.source_label,
                        "temperature_c": operation.temperature_c.midpoint if operation.temperature_c else None,
                        "time_h": operation.time_h.midpoint if operation.time_h else None,
                        "atmosphere": operation.atmosphere,
                    }
                    for operation in state.operations
                ],
            },
            "hard_checks": {
                "valid": hard_checks.valid,
                "flags": list(hard_checks.flags),
                "blocking_flags": list(hard_checks.blocking_flags),
                "notes": list(hard_checks.notes),
                "coverage_fraction": hard_checks.coverage_fraction,
                "reaction_balance": {
                    "feasible": hard_checks.reaction_balance.feasible if hard_checks.reaction_balance else False,
                    "equation": hard_checks.reaction_balance.equation if hard_checks.reaction_balance else None,
                    "environmental_reactants": [
                        {"formula": species.formula, "coefficient": species.coefficient}
                        for species in (hard_checks.reaction_balance.environmental_reactants if hard_checks.reaction_balance else ())
                    ],
                    "byproducts": [
                        {"formula": species.formula, "coefficient": species.coefficient}
                        for species in (hard_checks.reaction_balance.byproducts if hard_checks.reaction_balance else ())
                    ],
                    "unused_precursors": list(hard_checks.reaction_balance.unused_precursors if hard_checks.reaction_balance else ()),
                },
                "redox": {
                    "required_direction": hard_checks.redox.required_direction if hard_checks.redox else "unknown",
                    "environment_support": hard_checks.redox.environment_support if hard_checks.redox else "unknown",
                    "notes": list(hard_checks.redox.notes if hard_checks.redox else ()),
                    "flags": list(hard_checks.redox.flags if hard_checks.redox else ()),
                },
            },
            "retrieved_analogs": analog_payload,
        }


def build_judge(name: str, config: dict[str, Any] | None = None) -> BaseJudge:
    if name == "none":
        return NullJudge(config)
    if name == "deterministic":
        return DeterministicJudge(config)
    if name == "openai_structured":
        return OpenAICompatibleStructuredJudge(config)
    raise ValueError(f"Unknown judge: {name}")


def _clamp(value: float) -> float:
    return max(0.0, min(float(value), 1.0))
