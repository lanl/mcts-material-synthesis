#!/usr/bin/env python3
"""Run the leakage-controlled retro benchmark and persist results, then print.

Saves the JSON BEFORE printing so a formatting bug can never discard the compute.
"""
import argparse
import json
import os
import sys

from synthesis_planner.retro_benchmark import run_retro_benchmark


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", default="solid_state")
    ap.add_argument("--split-type", default="chemical_system")
    ap.add_argument("--max-targets", type=int, default=60)
    ap.add_argument("--iterations", type=int, default=60)
    ap.add_argument("--rollout-count", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--min-frequency", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = run_retro_benchmark(
        "data/processed", modality=args.modality, split_type=args.split_type,
        max_targets=args.max_targets, iterations=args.iterations,
        rollout_count=args.rollout_count, top_k=args.top_k,
        min_frequency=args.min_frequency, seed=args.seed,
    )

    os.makedirs("benchmark_results", exist_ok=True)
    out = args.out or f"benchmark_results/retro_{args.modality}_{args.split_type}.json"
    with open(out, "w") as handle:
        json.dump(res, handle, indent=2)

    a = res["_leakage_audit"]
    print(f"saved {out}")
    print(f"LEAKAGE: {'CLEAN' if a['clean'] else 'LEAK n=' + str(a['n_leaked'])} | "
          f"targets={res['_n_targets']} | unparseable_skipped={res['_n_unparseable_targets_skipped']}")
    cols = ["solve_rate", "mean_stock_coverage", "precursor_recall_at_1",
            "precursor_recall_at_k", "class_recall_at_k", "mean_precursor_jaccard",
            "condition_in_range_rate"]
    hdr = ["method", "solve", "cov", "rec@1", f"rec@{args.top_k}", "cls@k", "jacc", "cond"]
    print("".join(f"{h:>8s}" if i else f"{h:<18s}" for i, h in enumerate(hdr)))
    for method, r in res["methods"].items():
        row = f"{method:<18s}" + "".join(f"{r[c]:8.2f}" for c in cols)
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
