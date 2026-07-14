"""Pluggable retrieval-grounded route judges."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from .formula import safe_required_target_elements
from .schema import HardCheckResult, JudgeResult, PlanningState, RouteRecord


class BaseJudge:
    name = "base"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def evaluate(self, state: PlanningState, analogs: list[tuple[float, RouteRecord]], hard_checks: HardCheckResult) -> JudgeResult:
        raise NotImplementedError

    def evaluate_partial(self, state: PlanningState, analogs: list[tuple[float, RouteRecord]]) -> JudgeResult:
        """
        Evaluate incomplete route and flag likely missing steps.

        Override in subclasses to provide stage-specific checks.
        Default implementation returns neutral result.

        Args:
            state: Partial planning state
            analogs: Retrieved analogous routes

        Returns:
            JudgeResult with flags for likely issues
        """
        return JudgeResult(
            score=0.5,
            notes=("Partial state evaluation not implemented for this judge.",),
            flags=(),
            uncertainty=0.5,
        )


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

    def evaluate_partial(self, state: PlanningState, analogs: list[tuple[float, RouteRecord]]) -> JudgeResult:
        """
        Evaluate incomplete route and flag likely missing steps.

        Checks stage-specific issues:
        - precursors: element coverage, oxidant/reductant needs
        - preparation: mixing requirements
        - heating: temperature sufficiency
        """
        notes = []
        flags = []
        score = 0.5

        if state.stage == "precursors":
            # After precursor selection, check coverage and redox needs
            precursor_elements = set()
            for p in state.precursors:
                precursor_elements.update(p.elements)

            target_elements = set(safe_required_target_elements(state.problem.target_formula))
            if not target_elements.issubset(precursor_elements):
                missing = target_elements - precursor_elements
                flags.append("missing_element_source")
                notes.append(f"Precursor set does not cover all target elements: {', '.join(missing)}")
                score = 0.2

            # Check for redox needs
            if state.target_class in {"oxide", "sulfide", "nitride"}:
                has_oxidant = any("NO3" in p.formula or "O" in p.elements for p in state.precursors)
                if state.target_class == "oxide" and not has_oxidant:
                    flags.append("potential_oxidant_need")
                    notes.append("Oxide target may need oxidizing precursor or atmosphere.")
                    score *= 0.8

                if state.target_class in {"sulfide", "nitride"}:
                    # Check for reducing environment need
                    notes.append(f"{state.target_class.capitalize()} target likely needs reducing atmosphere or precursors.")

        elif state.stage == "preparation":
            # After preparation, check if operations are sufficient
            ops = [operation.verb for operation in state.operations]
            if "mix" not in ops and len(state.precursors) > 1:
                flags.append("missing_mixing")
                notes.append("Multiple precursors without mixing step.")
                score = 0.4

        elif state.stage == "heating":
            # After heating, check for regrinding needs
            if state.problem.modality == "solid_state":
                ops = [operation.verb for operation in state.operations]
                n_elements = len(safe_required_target_elements(state.problem.target_formula))

                if n_elements >= 3 and not any("grind" in op.lower() or "mill" in op.lower() for op in ops):
                    flags.append("potential_regrind_need")
                    notes.append("Multicomponent solid-state route may benefit from regrinding.")
                    score = 0.6

        elif state.stage == "finalize":
            # Check for modality-specific post-processing
            if state.problem.modality in {"hydrothermal", "precipitation"}:
                ops = [operation.verb for operation in state.operations]
                if "wash" not in ops or "dry" not in ops:
                    flags.append("incomplete_postprocessing")
                    notes.append("Solution-based route typically needs wash and dry steps.")
                    score = 0.5

        if not flags:
            notes.append("Partial route appears reasonable for current stage.")
            score = 0.7

        return JudgeResult(
            score=score,
            notes=tuple(notes),
            flags=tuple(flags),
            uncertainty=0.6,  # Partial states have higher uncertainty
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
        try:
            payload = self._request_structured_judgment(state, analogs, hard_checks)
        except Exception as exc:
            fallback = DeterministicJudge(self.config).evaluate(state, analogs, hard_checks)
            notes = list(fallback.notes)
            notes.append(f"Model-backed judge fallback used after structured output failure: {type(exc).__name__}.")
            flags = tuple(dict.fromkeys(fallback.flags + ("model_judge_fallback",)))
            return JudgeResult(
                score=fallback.score,
                notes=tuple(notes),
                flags=flags,
                evidence_dois=fallback.evidence_dois,
                rubric_scores=fallback.rubric_scores,
                uncertainty=max(fallback.uncertainty, 0.6),
            )
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
        api_style = self.config.get("api_style", "auto")
        payload = None

        if api_style in {"responses", "auto"}:
            try:
                payload = self._request_via_responses(client, model, state, analogs, hard_checks)
            except Exception:
                if api_style != "auto":
                    raise

        if payload is None and api_style in {"chat_completions", "auto"}:
            payload = self._request_via_chat_completions(client, model, state, analogs, hard_checks)

        if not isinstance(payload, dict):
            raise ValueError("Structured judge response was not a JSON object.")
        return _normalize_judge_payload(payload)

    def _request_via_responses(self, client, model: str, state: PlanningState, analogs, hard_checks: HardCheckResult) -> dict[str, Any]:
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
        return json.loads(response.output_text)

    def _request_via_chat_completions(self, client, model: str, state: PlanningState, analogs, hard_checks: HardCheckResult) -> dict[str, Any]:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": self._instructions() + " Return JSON only."},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "schema": self._schema,
                            "route_context": self._build_context_payload(state, analogs, hard_checks),
                        },
                        indent=2,
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Chat completions judge returned empty content.")
        return _parse_json_with_repair(content)

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
        # Check for prompt variant in config
        variant = self.config.get("prompt_variant")
        if variant and variant in EnsembleJudge.PROMPT_VARIANTS:
            return EnsembleJudge.PROMPT_VARIANTS[variant] + (
                " Review the proposed route using the supplied target, route chemistry, "
                "hard-check analysis, and retrieved literature analogs. "
                "Return valid JSON matching the schema exactly. "
                "Use evidence_dois only from the retrieved analog list."
            )

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
    if name == "ensemble":
        return EnsembleJudge(config)
    raise ValueError(f"Unknown judge: {name}")


def _clamp(value: float) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(float(value), 1.0))


def _normalize_judge_payload(payload: dict[str, Any]) -> dict[str, Any]:
    rubric = payload.get("rubric_scores") or {}
    notes = payload.get("notes") or []
    flags = payload.get("flags") or []
    evidence_dois = payload.get("evidence_dois") or []
    if isinstance(notes, str):
        notes = [notes]
    if isinstance(flags, str):
        flags = [flags]
    if isinstance(evidence_dois, str):
        evidence_dois = [evidence_dois]
    return {
        "score": _clamp(payload.get("score", 0.0)),
        "notes": [str(item) for item in notes],
        "flags": [str(item) for item in flags],
        "evidence_dois": [str(item) for item in evidence_dois],
        "rubric_scores": {
            "precursor_plausibility": _clamp(rubric.get("precursor_plausibility", 0.0)),
            "condition_compatibility": _clamp(rubric.get("condition_compatibility", 0.0)),
            "operation_completeness": _clamp(rubric.get("operation_completeness", 0.0)),
            "literature_analogy": _clamp(rubric.get("literature_analogy", 0.0)),
            "practicality": _clamp(rubric.get("practicality", 0.0)),
        },
        "uncertainty": _clamp(payload.get("uncertainty", 1.0)),
    }


def _parse_json_with_repair(content: str) -> dict[str, Any]:
    candidate = content.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = candidate[start : end + 1]

    for variant in (candidate, _repair_json(candidate)):
        try:
            payload = json.loads(variant)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("Chat completions judge returned malformed JSON that could not be repaired.")


def _repair_json(candidate: str) -> str:
    repaired = candidate
    repaired = re.sub(r':\s*null\{\s*"', ': null,\n  "', repaired)
    repaired = re.sub(r':\s*true\{\s*"', ': true,\n  "', repaired)
    repaired = re.sub(r':\s*false\{\s*"', ': false,\n  "', repaired)
    repaired = re.sub(r'(\bnull|\btrue|\bfalse|\d|\]|\}|")\s*\n\s*(")', r'\1,\n  \2', repaired)
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired


class EnsembleJudge(BaseJudge):
    """
    Ensemble judge that evaluates routes with multiple prompt variants.

    Uses disagreement across variants as an uncertainty proxy.
    """
    name = "ensemble"

    PROMPT_VARIANTS = {
        "conservative": (
            "You are a skeptical materials chemist reviewing a synthesis route. "
            "Identify potential failure modes and chemistry risks. "
            "Penalize unsupported novelty heavily. "
            "Default to uncertainty if evidence is weak. "
            "Be conservative in your scoring."
        ),
        "optimistic": (
            "You are an experienced synthesis chemist with practical lab intuition. "
            "Recognize when routes are plausible even if not exact literature matches. "
            "Give credit for reasonable adaptations and analogous chemistry. "
            "Be optimistic about feasibility when the chemistry is sound."
        ),
        "skeptical": (
            "You are an adversarial reviewer whose goal is to find flaws. "
            "What could go wrong? What steps are missing? What assumptions are questionable? "
            "Challenge the route's validity, completeness, and practicality. "
            "Be skeptical about success unless strongly supported by evidence."
        ),
    }

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.num_variants = config.get("num_variants", 3) if config else 3

        # Determine base judge type
        base_judge_name = config.get("base_judge", "deterministic") if config else "deterministic"

        if base_judge_name == "openai_structured":
            # Create variant judges with different prompts
            self.variant_judges = []
            for variant_name in list(self.PROMPT_VARIANTS.keys())[:self.num_variants]:
                variant_config = dict(config) if config else {}
                variant_config["prompt_variant"] = variant_name
                self.variant_judges.append(
                    OpenAICompatibleStructuredJudge(variant_config)
                )
        else:
            # For deterministic judge, create multiple instances with slight variations
            # (they'll be identical, but we track them for consistency)
            self.variant_judges = [
                build_judge(base_judge_name, config)
                for _ in range(self.num_variants)
            ]

    def evaluate(
        self,
        state: PlanningState,
        analogs: list[tuple[float, RouteRecord]],
        hard_checks: HardCheckResult
    ) -> JudgeResult:
        """
        Evaluate route with multiple judge variants and compute uncertainty.

        Returns ensemble mean with disagreement as uncertainty metric.
        """
        # Collect results from all variants
        results = []
        for judge in self.variant_judges:
            try:
                result = judge.evaluate(state, analogs, hard_checks)
                results.append(result)
            except Exception as e:
                # If one variant fails, continue with others
                print(f"Warning: Ensemble variant failed: {e}")
                continue

        if not results:
            # All variants failed, return null result
            return JudgeResult(
                score=0.0,
                notes=("All ensemble variants failed.",),
                flags=("ensemble_failure",),
                uncertainty=1.0,
            )

        # Compute ensemble statistics
        scores = [r.score for r in results]
        mean_score = sum(scores) / len(scores)

        # Disagreement as uncertainty: max - min score
        disagreement = max(scores) - min(scores) if len(scores) > 1 else 0.0

        # Variance as additional uncertainty signal
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_dev = variance ** 0.5

        # Combined uncertainty (disagreement weighted more heavily)
        uncertainty = min(1.0, 0.7 * disagreement + 0.3 * std_dev)

        # Merge notes and flags from all variants
        all_notes = []
        all_flags = []
        for result in results:
            all_notes.extend(result.notes)
            all_flags.extend(result.flags)

        # Deduplicate while preserving order
        seen_notes = set()
        unique_notes = []
        for note in all_notes:
            if note not in seen_notes:
                seen_notes.add(note)
                unique_notes.append(note)

        unique_flags = list(dict.fromkeys(all_flags))  # Deduplicate flags

        # Merge evidence DOIs (take union)
        all_dois = set()
        for result in results:
            all_dois.update(result.evidence_dois)

        # Average rubric scores across variants
        rubric_keys = set()
        for result in results:
            rubric_keys.update(result.rubric_scores.keys())

        merged_rubric = {}
        for key in rubric_keys:
            values = [r.rubric_scores.get(key, 0.0) for r in results if key in r.rubric_scores]
            if values:
                merged_rubric[key] = sum(values) / len(values)

        return JudgeResult(
            score=mean_score,
            notes=tuple(unique_notes[:10]),  # Limit to top 10 notes
            flags=tuple(unique_flags),
            evidence_dois=tuple(sorted(all_dois)),
            rubric_scores=merged_rubric,
            uncertainty=uncertainty,
        )
