from synthesis_planner.cli import build_parser, load_config


def test_load_config_missing_file_returns_empty_dict(tmp_path):
    assert load_config(str(tmp_path / "missing.json")) == {}


def test_load_config_reads_python_config_file(tmp_path):
    config_path = tmp_path / "config.py"
    config_path.write_text("CONFIG = {'judge': {'name': 'openai_structured', 'model': 'gpt-4o-mini'}}\n")
    config = load_config(str(config_path))
    assert config["judge"]["name"] == "openai_structured"
    assert config["judge"]["model"] == "gpt-4o-mini"


def test_plan_subcommand_parses_target():
    parser = build_parser({})
    args = parser.parse_args(["plan", "--target", "BaTiO3", "--max-temperature-c", "1200"])
    assert args.command == "plan"
    assert args.target == "BaTiO3"
    assert args.modality == "solid_state"
    assert args.max_temperature_c == 1200.0


def test_download_data_subcommand_parses():
    parser = build_parser({})
    args = parser.parse_args(["download-data"])
    assert args.command == "download-data"


def test_benchmark_subcommand_parses():
    parser = build_parser({})
    args = parser.parse_args(["benchmark", "--split-type", "chemical_system", "--method", "nearest_neighbor"])
    assert args.command == "benchmark"
    assert args.split_type == "chemical_system"
    assert args.method == "nearest_neighbor"


def test_plan_subcommand_parses_openai_judge_arguments():
    parser = build_parser({})
    args = parser.parse_args(
        [
            "plan",
            "--target",
            "BaTiO3",
            "--judge",
            "openai_structured",
            "--judge-model",
            "gpt-4o-mini",
            "--judge-api-key",
            "test-key",
            "--judge-base-url",
            "https://api.openai.com/v1",
        ]
    )
    assert args.judge == "openai_structured"
    assert args.judge_model == "gpt-4o-mini"
    assert args.judge_api_key == "test-key"
    assert args.judge_base_url == "https://api.openai.com/v1"
