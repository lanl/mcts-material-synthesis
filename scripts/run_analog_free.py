#!/usr/bin/env python3
"""Analog-free vs analog-rich stratified benchmark (saves JSON before printing)."""
import argparse
import json
import os
import sys

from synthesis_planner.retro_benchmark import run_analog_free_benchmark

COLS = ["mean_synthesizability", "mean_operation_similarity", "condition_in_range_rate",
        "stage_count_match_rate", "solve_rate", "precursor_recall_at_1", "mean_precursor_jaccard"]
HDR = ["method", "synth", "op_sim", "cond", "stage", "solve", "rec@1", "jacc"]


def _print_stratum(name, block):
    s = block["_stratum"]
    print(f"\n[{name}] n={s['n']} support[min={s['support_min']:.2f} "
          f"mean={s['support_mean']:.2f} max={s['support_max']:.2f}]")
    print("".join(f"{h:>8s}" if i else f"{h:<18s}" for i, h in enumerate(HDR)))
    for method, r in block["methods"].items():
        print(f"{method:<18s}" + "".join(f"{r[c]:8.2f}" for c in COLS))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", default="solid_state")
    ap.add_argument("--split-type", default="chemical_system")
    ap.add_argument("--support-quantile", type=float, default=0.33)
    ap.add_argument("--max-targets-per-stratum", type=int, default=60)
    ap.add_argument("--iterations", type=int, default=60)
    ap.add_argument("--rollout-count", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--min-frequency", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    res = run_analog_free_benchmark(
        "data/processed", modality=args.modality, split_type=args.split_type,
        support_quantile=args.support_quantile,
        max_targets_per_stratum=args.max_targets_per_stratum,
        iterations=args.iterations, rollout_count=args.rollout_count,
        top_k=args.top_k, min_frequency=args.min_frequency, seed=args.seed,
    )
    os.makedirs("benchmark_results", exist_ok=True)
    out = f"benchmark_results/analog_free_{args.modality}_{args.split_type}.json"
    with open(out, "w") as handle:
        json.dump(res, handle, indent=2)
    print(f"saved {out}")
    a = res["_leakage_audit"]
    print(f"LEAKAGE: {'CLEAN' if a['clean'] else 'LEAK'} | parseable_test_targets={res['_n_test_targets_parseable']}")
    _print_stratum("ANALOG-FREE (novel)", res["analog_free"])
    _print_stratum("ANALOG-RICH", res["analog_rich"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
