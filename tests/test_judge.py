from synthesis_planner.formula import infer_target_class, parse_formula
from synthesis_planner.judge import _parse_json_with_repair, build_judge
from synthesis_planner.schema import HardCheckResult, PlanningProblem, PlanningState, PrecursorRecord, ReactionBalanceResult, RedoxAnalysisResult, RouteRecord


def _state_for():
    problem = PlanningProblem(target_formula="BaTiO3")
    return PlanningState(
        problem=problem,
        target_elements=tuple(sorted(parse_formula(problem.target_formula))),
        target_class=infer_target_class(problem.target_formula),
        stage="terminal",
        precursors=(
            PrecursorRecord("BaCO3", "carbonate", ("Ba", "C", "O")),
            PrecursorRecord("TiO2", "oxide", ("Ti", "O")),
        ),
    )


def _hard_checks():
    return HardCheckResult(
        valid=True,
        flags=(),
        notes=(),
        coverage_fraction=1.0,
        blocking_flags=(),
        reaction_balance=ReactionBalanceResult(
            feasible=True,
            framework_match_fraction=1.0,
            precursor_coefficients=(1.0, 1.0),
            equation="1 BaCO3 + 1 TiO2 -> BaTiO3 + 1 CO2",
        ),
        redox=RedoxAnalysisResult(
            target_charge=6.0,
            precursor_charge=6.0,
            required_direction="none",
            environment_support="supported",
            notes=(),
            flags=(),
        ),
    )


def _analogs():
    route = RouteRecord(
        route_id="route-1",
        source_doi="10.1000/example",
        publication_year=2020,
        modality="solid_state",
        target_formula="BaTiO3",
        target_elements=("Ba", "O", "Ti"),
        chemical_system="Ba-O-Ti",
        target_class="oxide",
        precursors=(
            PrecursorRecord("BaCO3", "carbonate", ("Ba", "C", "O")),
            PrecursorRecord("TiO2", "oxide", ("Ti", "O")),
        ),
        solvents=(),
        operations=(),
        reaction_string="BaCO3 + TiO2 -> BaTiO3",
        paragraph_excerpt="A representative solid-state route.",
        source_dataset="test",
    )
    return [(8.5, route)]


def test_openai_structured_judge_uses_model_payload(monkeypatch):
    judge = build_judge(
        "openai_structured",
        {"model": "gpt-4o-mini", "api_key": "test-key"},
    )

    class _FakeResponses:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return type(
                "Response",
                (),
                {
                    "output_text": (
                        '{"score": 0.82, "notes": ["Analog route support is strong."], '
                        '"flags": ["requires_validation"], "evidence_dois": ["10.1000/example"], '
                        '"rubric_scores": {"precursor_plausibility": 0.9, "condition_compatibility": 0.8, '
                        '"operation_completeness": 0.7, "literature_analogy": 0.95, "practicality": 0.75}, '
                        '"uncertainty": 0.2}'
                    )
                },
            )()

    fake_responses = _FakeResponses()
    fake_client = type("Client", (), {"responses": fake_responses})()
    monkeypatch.setattr(judge, "_build_client", lambda: fake_client)

    result = judge.evaluate(_state_for(), _analogs(), _hard_checks())
    assert result.score == 0.82
    assert result.flags == ("requires_validation",)
    assert result.evidence_dois == ("10.1000/example",)
    assert fake_responses.kwargs["model"] == "gpt-4o-mini"
    assert fake_responses.kwargs["text"]["format"]["type"] == "json_schema"


def test_openai_structured_judge_falls_back_to_chat_completions(monkeypatch):
    judge = build_judge(
        "openai_structured",
        {"model": "gpt-oss-120b", "api_key": "test-key", "api_style": "auto"},
    )

    class _FakeResponses:
        def create(self, **kwargs):
            raise RuntimeError("responses endpoint unavailable")

    class _FakeChatCompletions:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return type(
                "Response",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {
                                "message": type(
                                    "Message",
                                    (),
                                    {
                                        "content": (
                                            '{"score": 0.7, "notes": ["Proxy route judged successfully."], '
                                            '"flags": [], "evidence_dois": ["10.1000/example"], '
                                            '"rubric_scores": {"precursor_plausibility": 0.8, "condition_compatibility": 0.7, '
                                            '"operation_completeness": 0.6, "literature_analogy": 0.9, "practicality": 0.65}, '
                                            '"uncertainty": 0.3}'
                                        )
                                    },
                                )()
                            },
                        )()
                    ]
                },
            )()

    fake_chat = _FakeChatCompletions()
    fake_client = type(
        "Client",
        (),
        {"responses": _FakeResponses(), "chat": type("Chat", (), {"completions": fake_chat})()},
    )()
    monkeypatch.setattr(judge, "_build_client", lambda: fake_client)

    result = judge.evaluate(_state_for(), _analogs(), _hard_checks())
    assert result.score == 0.7
    assert fake_chat.kwargs["model"] == "gpt-oss-120b"
    assert fake_chat.kwargs["response_format"]["type"] == "json_object"


def test_parse_json_with_repair_handles_proxy_malformed_null():
    payload = _parse_json_with_repair(
        '{\n  "score": null{\n  "notes": "Insufficient information.",\n  "flags": ["missing_input"],\n  "evidence_dois": [],\n  "rubric_scores": {},\n  "uncertainty": 0.9\n}'
    )
    assert payload["score"] is None
    assert payload["flags"] == ["missing_input"]


def test_openai_structured_judge_falls_back_to_deterministic_on_parse_failure(monkeypatch):
    judge = build_judge(
        "openai_structured",
        {"model": "gpt-oss-120b", "api_key": "test-key", "api_style": "chat_completions"},
    )
    monkeypatch.setattr(judge, "_request_structured_judgment", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad json")))
    result = judge.evaluate(_state_for(), _analogs(), _hard_checks())
    assert "model_judge_fallback" in result.flags
    assert any("structured output failure" in note for note in result.notes)
