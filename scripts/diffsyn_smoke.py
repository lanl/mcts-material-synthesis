"""Phase-2 smoke test: real frozen DiffSyn -> our discovery search, on CPU.

Validates that ``DiffSynConditionGenerator`` loads the pretrained checkpoint and
that swapping it in for ``StubConditionGenerator`` needs *zero* other changes.

Run with the torch env (which has torch/einops/sklearn/torchvision) and the repo
``src`` on the path, pointing at a populated ``zeosyn_gen`` clone:

    PYTHONPATH=src /p/lustre2/laubach2/diffsyn-venv/bin/python \
        scripts/diffsyn_smoke.py --repo-root /path/to/zeosyn_gen

Prerequisites in the clone (see DISCOVERY_DIFFSYN_INTEGRATION.md Phase 2):
  runs/diff/system/run1/{configs.json, model.pt(from Dropbox)},
  data/{zeolite_descriptors.csv, ZeoSynGen_dataset.pkl, scalers/*.pkl,
  syn_variables.py, an OSDA feature CSV}, models/diffusion.py.

Defaults use the paper's UFI + K222 demo pair (guaranteed present in the OSDA CSV).
"""

from __future__ import annotations

import argparse

from synthesis_planner.discovery import (
    DiffSynConditionGenerator,
    DiffSynDiscoverySearch,
    OSDA,
    SearchConfig,
    ZeoliteNoveltyIndex,
    ZeoliteTarget,
)
from synthesis_planner.discovery.oracle import BindingEnergyOracle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True, help="path to a populated zeosyn_gen clone")
    ap.add_argument("--zeolite", default="UFI")
    ap.add_argument(
        "--osda", default="C1COCCN2CCOCCOCCN(CCO1)CCOCCOCC2", help="OSDA SMILES (must be in the CSV)"
    )
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--sampling-timesteps", type=int, default=250,
                    help="DDIM steps. CPU-tolerable; NOTE <~100 gives degenerate samples "
                         "(Si/Al,OH/T,sda1/T collapse) on this DDPM-tuned model. Use ~1000 "
                         "(DDPM) or a GPU for faithful values.")
    ap.add_argument("--threads", type=int, default=4, help="cap torch intra-op threads")
    args = ap.parse_args()

    gen = DiffSynConditionGenerator(
        repo_root=args.repo_root, device="cpu",
        sampling_timesteps=args.sampling_timesteps, num_threads=args.threads,
    )
    target, osda = ZeoliteTarget(args.zeolite), OSDA(smiles=args.osda)

    print(f"[1] Sampling {args.n} conditions for {args.zeolite} + OSDA via frozen DiffSyn (CPU)...")
    samples = gen.sample(target, osda, args.n, seed=0)
    assert len(samples) == args.n
    v0 = samples[0]
    assert len(v0.as_tuple()) == 12, "expected the exact 12-variable DiffSyn contract"
    print("    first sample (physical units):")
    for name in v0.spec.variables:
        print(f"      {name:10s} = {v0.get(name):.3f}")

    print("[2] Plugging the SAME generator into DiffSynDiscoverySearch (stub -> real, no other change)...")
    search = DiffSynDiscoverySearch(
        generator=gen,
        oracle=BindingEnergyOracle(None),  # neutral until a real binding-energy fn is wired
        novelty_index=ZeoliteNoveltyIndex(reference={}),  # empty corpus -> pure feasibility+realism
        osda_pool=[osda],
        config=SearchConfig(iterations=40, top_k=3, seed=0),
    )
    portfolio = search.run(target)
    print(f"    portfolio: {len(portfolio)} recipe(s)")
    for i, r in enumerate(portfolio):
        rb = search.reward_of(r, target)
        print(
            f"      #{i}: T={r.conditions.get('cryst_temp'):.0f}C  "
            f"Si/Al={r.conditions.get('Si/Al'):.1f}  "
            f"reward={rb.total:.3f} (feas={rb.feasibility:.2f} nov={rb.novelty:.2f} real={rb.realism:.2f})"
        )
    print("SMOKE OK")


if __name__ == "__main__":
    main()
