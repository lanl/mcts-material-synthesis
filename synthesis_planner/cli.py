"""CLI for downloading data, preparing indices, and planning routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .benchmark import build_split, evaluate_split, load_solid_state_routes, save_benchmark_summary, save_split_manifest
from .datasets import download_public_datasets, prepare_processed_data
from .planner import SynthesisPlanner
from .schema import LabConstraints, PlanningProblem


def load_config(config_path: str = "config.json") -> dict:
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def build_parser(config: dict | None = None) -> argparse.ArgumentParser:
    config = config or {}
    parser = argparse.ArgumentParser(description="Target-conditioned MCTS synthesis planning")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download-data", help="Download the public synthesis datasets")
    download.add_argument("--data-dir", default=config.get("data_dir", "data/raw"))

    prepare = subparsers.add_parser("prepare-data", help="Normalize the raw datasets into route records")
    prepare.add_argument("--data-dir", default=config.get("data_dir", "data/raw"))
    prepare.add_argument("--processed-dir", default=config.get("processed_dir", "data/processed"))

    plan = subparsers.add_parser("plan", help="Plan solid-state synthesis routes for a target")
    plan.add_argument("--target", required=True, help="Target formula, for example BaTiO3")
    plan.add_argument("--modality", choices=["solid_state"], default=config.get("modality", "solid_state"))
    plan.add_argument("--data-dir", default=config.get("data_dir", "data/raw"))
    plan.add_argument("--processed-dir", default=config.get("processed_dir", "data/processed"))
    plan.add_argument("--iterations", type=int, default=config.get("iterations", 250))
    plan.add_argument("--top-k", type=int, default=config.get("top_k", 5))
    plan.add_argument("--exploration-constant", type=float, default=config.get("exploration_constant", 1.4))
    plan.add_argument("--rollout-count", type=int, default=config.get("rollout_count", 8))
    plan.add_argument("--seed", type=int, default=config.get("seed"))
    plan.add_argument("--min-temperature-c", type=float, default=None)
    plan.add_argument("--max-temperature-c", type=float, default=None)
    plan.add_argument("--allowed-atmosphere", action="append", default=[])
    plan.add_argument("--forbid-precursor-class", action="append", default=[])
    plan.add_argument("--output-dir", default="planning_results")

    make_splits = subparsers.add_parser("make-splits", help="Create benchmark split manifests from processed solid-state routes")
    make_splits.add_argument("--processed-dir", default=config.get("processed_dir", "data/processed"))
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
    benchmark.add_argument(
        "--split-type",
        choices=["random", "target_formula", "chemical_system", "material_family", "publication_year"],
        default="target_formula",
    )
    benchmark.add_argument("--test-fraction", type=float, default=0.2)
    benchmark.add_argument("--iterations", type=int, default=50)
    benchmark.add_argument("--top-k", type=int, default=1)
    benchmark.add_argument("--rollout-count", type=int, default=3)
    benchmark.add_argument("--seed", type=int, default=config.get("seed"))
    benchmark.add_argument("--output", default="benchmark_results/summary.json")

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
        planner = SynthesisPlanner(args.data_dir, args.processed_dir)
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
        )
        if not routes:
            print("No routes generated.")
            return 1

        output_path = planner.save_routes(routes, args.output_dir)
        for idx, route in enumerate(routes, start=1):
            precursors = ", ".join(precursor.formula for precursor in route.precursors)
            operations = " -> ".join(operation.source_label or operation.verb for operation in route.operations)
            print(f"Rank {idx}: score={route.score.total:.3f}")
            print(f"  Precursors: {precursors}")
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
        routes = load_solid_state_routes(args.processed_dir)
        train_routes, test_routes = build_split(routes, args.split_type, test_fraction=args.test_fraction, seed=args.seed)
        output = save_split_manifest(train_routes, test_routes, args.split_type, args.output)
        print(f"train: {len(train_routes)}")
        print(f"test: {len(test_routes)}")
        print(f"saved: {output}")
        return 0

    if args.command == "benchmark":
        planner = SynthesisPlanner(processed_dir=args.processed_dir)
        routes = load_solid_state_routes(args.processed_dir)
        train_routes, test_routes = build_split(routes, args.split_type, test_fraction=args.test_fraction, seed=args.seed)
        summary = evaluate_split(
            planner,
            train_routes,
            test_routes,
            split_type=args.split_type,
            iterations=args.iterations,
            top_k=args.top_k,
            rollout_count=args.rollout_count,
            seed=args.seed,
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

    return 1
