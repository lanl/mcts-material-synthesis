"""CLI for downloading data, preparing indices, and planning routes."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

from .benchmark import build_split, evaluate_method_suite, evaluate_split, load_routes, save_benchmark_summary, save_split_manifest
from .datasets import download_public_datasets, prepare_processed_data
from .judge_calibration import calibrate_judge, print_calibration_report
from .materials_project import create_mp_client_from_config
from .planner import SynthesisPlanner
from .schema import LabConstraints, PlanningProblem


def load_config(config_path: str | None = None) -> dict:
    if config_path is not None:
        return _load_config_file(Path(config_path))

    config = {}
    for candidate in (Path("config.py"), Path("config.json")):
        config = _deep_merge(config, _load_config_file(candidate))
    return config


def _load_config_file(path: Path) -> dict:
    if not path.exists():
        return {}
    if path.suffix == ".py":
        return _load_python_config(path)
    if path.suffix == ".json":
        return _load_json_config(path)
    return {}


def _load_python_config(path: Path) -> dict:
    spec = importlib.util.spec_from_file_location("mcts_material_synthesis_local_config", path)
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = getattr(module, "CONFIG", None)
    return config if isinstance(config, dict) else {}


def _load_json_config(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _deep_merge(base: dict, updates: dict) -> dict:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_parser(config: dict | None = None) -> argparse.ArgumentParser:
    config = config or {}
    judge_config = config.get("judge", {})
    parser = argparse.ArgumentParser(description="Target-conditioned MCTS synthesis planning")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download-data", help="Download the public synthesis datasets")
    download.add_argument("--data-dir", default=config.get("data_dir", "data/raw"))

    prepare = subparsers.add_parser("prepare-data", help="Normalize the raw datasets into route records")
    prepare.add_argument("--data-dir", default=config.get("data_dir", "data/raw"))
    prepare.add_argument("--processed-dir", default=config.get("processed_dir", "data/processed"))

    plan = subparsers.add_parser("plan", help="Plan solid-state synthesis routes for a target")
    plan.add_argument("--target", required=True, help="Target formula, for example BaTiO3")
    plan.add_argument("--modality", choices=["solid_state", "hydrothermal", "precipitation"], default=config.get("modality", "solid_state"))
    plan.add_argument("--data-dir", default=config.get("data_dir", "data/raw"))
    plan.add_argument("--processed-dir", default=config.get("processed_dir", "data/processed"))
    plan.add_argument("--iterations", type=int, default=config.get("iterations", 250))
    plan.add_argument("--top-k", type=int, default=config.get("top_k", 5))
    plan.add_argument("--exploration-constant", type=float, default=config.get("exploration_constant", 1.4))
    plan.add_argument("--rollout-count", type=int, default=config.get("rollout_count", 8))
    plan.add_argument("--seed", type=int, default=config.get("seed"))
    plan.add_argument("--judge", choices=["deterministic", "none", "openai_structured"], default=judge_config.get("name", "deterministic"))
    plan.add_argument("--judge-model", default=judge_config.get("model"))
    plan.add_argument("--judge-api-key", default=judge_config.get("api_key"))
    plan.add_argument("--judge-base-url", default=judge_config.get("base_url"))
    plan.add_argument("--judge-api-style", choices=["auto", "responses", "chat_completions"], default=judge_config.get("api_style", "auto"))
    plan.add_argument("--disable-judge", action="store_true")
    plan.add_argument("--disable-hard-checks", action="store_true")
    plan.add_argument("--disable-retrieval", action="store_true")
    plan.add_argument("--min-temperature-c", type=float, default=None)
    plan.add_argument("--max-temperature-c", type=float, default=None)
    plan.add_argument("--allowed-atmosphere", action="append", default=[])
    plan.add_argument("--forbid-precursor-class", action="append", default=[])
    plan.add_argument("--output-dir", default="planning_results")

    make_splits = subparsers.add_parser("make-splits", help="Create benchmark split manifests from processed solid-state routes")
    make_splits.add_argument("--processed-dir", default=config.get("processed_dir", "data/processed"))
    make_splits.add_argument("--modality", choices=["solid_state", "hydrothermal", "precipitation"], default=config.get("modality", "solid_state"))
    make_splits.add_argument(
        "--split-type",
        choices=["random", "target_formula", "chemical_system", "material_family", "publication_year"],
        default="target_formula",
    )
    make_splits.add_argument("--test-fraction", type=float, default=0.2)
    make_splits.add_argument("--seed", type=int, default=config.get("seed"))
    make_splits.add_argument("--output", default="benchmark_results/split_manifest.json")

    benchmark = subparsers.add_parser("benchmark", help="Run a retrospective benchmark over a held-out split")
    benchmark.add_argument("--processed-dir", default=config.get("processed_dir", "data/processed"))
    benchmark.add_argument("--modality", choices=["solid_state", "hydrothermal", "precipitation"], default=config.get("modality", "solid_state"))
    benchmark.add_argument(
        "--split-type",
        choices=["random", "target_formula", "chemical_system", "material_family", "publication_year"],
        default="target_formula",
    )
    benchmark.add_argument(
        "--method",
        choices=["mcts", "nearest_neighbor", "frequency_prior", "mcts_no_retrieval", "mcts_no_judge", "mcts_no_hard_checks", "suite"],
        default="mcts",
    )
    benchmark.add_argument("--test-fraction", type=float, default=0.2)
    benchmark.add_argument("--iterations", type=int, default=50)
    benchmark.add_argument("--top-k", type=int, default=1)
    benchmark.add_argument("--rollout-count", type=int, default=3)
    benchmark.add_argument("--seed", type=int, default=config.get("seed"))
    benchmark.add_argument("--judge", choices=["deterministic", "none", "openai_structured"], default=judge_config.get("name", "deterministic"))
    benchmark.add_argument("--judge-model", default=judge_config.get("model"))
    benchmark.add_argument("--judge-api-key", default=judge_config.get("api_key"))
    benchmark.add_argument("--judge-base-url", default=judge_config.get("base_url"))
    benchmark.add_argument("--judge-api-style", choices=["auto", "responses", "chat_completions"], default=judge_config.get("api_style", "auto"))
    benchmark.add_argument("--output", default="benchmark_results/summary.json")

    calibrate = subparsers.add_parser("calibrate-judge", help="Calibrate judge against held-out ground-truth routes")
    calibrate.add_argument("--processed-dir", default=config.get("processed_dir", "data/processed"))
    calibrate.add_argument("--modality", choices=["solid_state", "hydrothermal", "precipitation"], default="solid_state")
    calibrate.add_argument("--judge", choices=["deterministic", "none", "openai_structured"], default=judge_config.get("name", "deterministic"))
    calibrate.add_argument("--judge-model", default=judge_config.get("model"))
    calibrate.add_argument("--judge-api-key", default=judge_config.get("api_key"))
    calibrate.add_argument("--judge-base-url", default=judge_config.get("base_url"))
    calibrate.add_argument("--judge-api-style", choices=["auto", "responses", "chat_completions"], default=judge_config.get("api_style", "auto"))
    calibrate.add_argument("--split-type", choices=["random", "target_formula", "chemical_system"], default="target_formula")
    calibrate.add_argument("--test-fraction", type=float, default=0.2)
    calibrate.add_argument("--max-samples", type=int, default=100, help="Maximum test routes to evaluate")
    calibrate.add_argument("--seed", type=int, default=config.get("seed"))
    calibrate.add_argument("--output", default="calibration_report.json")

    return parser


def main(argv: list[str] | None = None) -> int:
    config = load_config()
    args = build_parser(config).parse_args(argv)

    if args.command == "download-data":
        destinations = download_public_datasets(args.data_dir)
        for key, path in destinations.items():
            print(f"{key}: {path}")
        return 0

    if args.command == "prepare-data":
        outputs = prepare_processed_data(args.data_dir, args.processed_dir)
        for key, path in outputs.items():
            print(f"{key}: {path}")
        return 0

    if args.command == "plan":
        mp_client = create_mp_client_from_config(config)
        planner = SynthesisPlanner(args.data_dir, args.processed_dir, mp_client=mp_client)
        judge_config = _build_judge_config(args)
        constraints = LabConstraints(
            min_temperature_c=args.min_temperature_c,
            max_temperature_c=args.max_temperature_c,
            allowed_atmospheres=tuple(args.allowed_atmosphere),
            forbidden_precursor_classes=tuple(args.forbid_precursor_class),
        )
        routes = planner.plan(
            PlanningProblem(target_formula=args.target, modality=args.modality, lab_constraints=constraints),
            iterations=args.iterations,
            top_k=args.top_k,
            exploration_constant=args.exploration_constant,
            rollout_count=args.rollout_count,
            seed=args.seed,
            judge_name=args.judge,
            judge_config=judge_config,
            use_judge=not args.disable_judge,
            use_hard_checks=not args.disable_hard_checks,
            use_retrieval=not args.disable_retrieval,
        )
        if not routes:
            print("No routes generated.")
            return 1

        output_path = planner.save_routes(routes, args.output_dir)
        for idx, route in enumerate(routes, start=1):
            precursors = ", ".join(precursor.formula for precursor in route.precursors)
            solvents = ", ".join(route.solvents) if route.solvents else "n/a"
            operations = " -> ".join(operation.source_label or operation.verb for operation in route.operations)
            print(f"Rank {idx}: score={route.score.total:.3f}")
            print(f"  Precursors: {precursors}")
            print(f"  Solvents: {solvents}")
            print(f"  Operations: {operations}")
            print(f"  Valid: {route.hard_checks.valid}")
            print(f"  Evidence: {', '.join(route.evidence_dois[:3]) if route.evidence_dois else 'n/a'}")
            if route.hard_checks.blocking_flags:
                print(f"  Hard flags: {', '.join(route.hard_checks.blocking_flags)}")
            if route.judge.notes:
                print(f"  Notes: {route.judge.notes[0]}")
        print(f"Saved: {output_path}")
        return 0

    if args.command == "make-splits":
        routes = load_routes(args.processed_dir, args.modality)
        train_routes, test_routes = build_split(routes, args.split_type, test_fraction=args.test_fraction, seed=args.seed)
        output = save_split_manifest(train_routes, test_routes, args.split_type, args.output)
        print(f"train: {len(train_routes)}")
        print(f"test: {len(test_routes)}")
        print(f"saved: {output}")
        return 0

    if args.command == "benchmark":
        mp_client = create_mp_client_from_config(config)
        planner = SynthesisPlanner(processed_dir=args.processed_dir, mp_client=mp_client)
        judge_config = _build_judge_config(args)
        routes = load_routes(args.processed_dir, args.modality)
        train_routes, test_routes = build_split(routes, args.split_type, test_fraction=args.test_fraction, seed=args.seed)
        if args.method == "suite":
            methods = ["mcts", "nearest_neighbor", "frequency_prior", "mcts_no_retrieval", "mcts_no_judge", "mcts_no_hard_checks"]
            report = evaluate_method_suite(
                planner,
                train_routes,
                test_routes,
                split_type=args.split_type,
                methods=methods,
                modality=args.modality,
                iterations=args.iterations,
                top_k=args.top_k,
                rollout_count=args.rollout_count,
                seed=args.seed,
                judge_name=args.judge,
                judge_config=judge_config,
            )
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2))
            for method, summary in report.items():
                print(f"{method}: validity={summary['top1_validity_rate']:.3f}, op_sim={summary['mean_operation_similarity']:.3f}")
            print(f"saved: {output_path}")
            return 0
        summary = evaluate_split(
            planner,
            train_routes,
            test_routes,
            split_type=args.split_type,
            method=args.method,
            modality=args.modality,
            iterations=args.iterations,
            top_k=args.top_k,
            rollout_count=args.rollout_count,
            seed=args.seed,
            judge_name=args.judge,
            judge_config=judge_config,
        )
        output = save_benchmark_summary(summary, args.output)
        print(f"split_type: {summary.split_type}")
        print(f"train: {summary.n_train}")
        print(f"test: {summary.n_test}")
        print(f"precursor_exact_match_at_1: {summary.precursor_exact_match_at_1:.3f}")
        print(f"precursor_class_match_at_1: {summary.precursor_class_match_at_1:.3f}")
        print(f"top1_validity_rate: {summary.top1_validity_rate:.3f}")
        print(f"mean_operation_similarity: {summary.mean_operation_similarity:.3f}")
        print(f"mean_temperature_error_c: {summary.mean_temperature_error_c}")
        print(f"saved: {output}")
        return 0

    if args.command == "calibrate-judge":
        routes = load_routes(args.processed_dir, args.modality)
        train_routes, test_routes = build_split(routes, args.split_type, test_fraction=args.test_fraction, seed=args.seed)

        judge_config = _build_judge_config(args)
        result = calibrate_judge(
            judge_name=args.judge,
            test_routes=test_routes,
            train_routes=train_routes,
            judge_config=judge_config,
            max_samples=args.max_samples,
        )

        # Print human-readable report
        print_calibration_report(result)

        # Save JSON report
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"Report saved to: {output_path}")
        return 0

    return 1


def _build_judge_config(args) -> dict:
    config = {}
    if getattr(args, "judge_model", None):
        config["model"] = args.judge_model
    if getattr(args, "judge_api_key", None):
        config["api_key"] = args.judge_api_key
    if getattr(args, "judge_base_url", None):
        config["base_url"] = args.judge_base_url
    if getattr(args, "judge_api_style", None):
        config["api_style"] = args.judge_api_style
    return config
