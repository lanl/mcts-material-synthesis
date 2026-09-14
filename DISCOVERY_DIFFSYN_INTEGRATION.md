# MCTS × DiffSyn: Generative Zeolite Synthesis Discovery (Strategy B)

**Status doc + handoff record.** If this session ended and you are resuming: read
this file top-to-bottom first, then the `STATUS` section at the very bottom for
exactly what is done, what is stubbed, and the next action. Everything in
`src/synthesis_planner/discovery/` is new and self-contained; the oxide pipeline
(`core/`, `benchmark.py`, …) is untouched.

---

## 1. The idea in one paragraph

Reframe the project from **retrospective recipe recall** (where MCTS ties/trails a
frequency prior — see `RESULTS_RETRO_BENCHMARK.md` Runs A–F) to **generative
synthesis discovery**. We adopt **Strategy B**: pivot the discovery target to
**zeolites**, reuse the **pretrained DiffSyn** conditional diffusion model
(Nature Comput. Sci. 2025, `s43588-025-00949-9`; code `eltonpan/zeosyn_gen`,
data `ZeoSyn`, figshare `10.6084/m9.figshare.30632942`) **out of the box, frozen,
no retraining**, and put **MCTS as the outer search loop** over the discrete
synthesis decisions DiffSyn conditions *on* but does not itself choose. DiffSyn
supplies realistic continuous conditions; MCTS supplies multi-objective search,
novelty-seeking, and a diverse portfolio; a feasibility oracle (DFT
OSDA–framework binding energy, per the paper) grounds it.

## 2. What DiffSyn is (verified from paper + repo)

- **Model:** classifier-free-guided **diffusion** model. U-Net denoiser; an
  **E(3)-equivariant GNN** encodes the zeolite framework; the **OSDA** enters as
  molecular descriptors. Denoises a vector of **continuous synthesis variables**.
- **I/O contract — CONFIRMED from source** (`eltonpan/zeosyn_gen`, checkpoint
  `runs/diff/system/run1`, verified 2026-09-14; this is the integration surface):
  - **Conditioning is descriptor vectors, not raw code/SMILES:**
    - zeolite → **143-dim descriptor vector** (`syn_variables.zeo_cols`: Zeo++
      pore/channel/surface geometry), looked up by IZA code from
      `data/zeolite_descriptors.csv`. (`zeo_feat_dims=143`.) NB: this released
      `diff/run1` model conditions on **descriptors**, not the E(3)-GNN/CIF the
      abstract mentions — the GNN is a `*-eq` variant. Simpler for us: no GNN/CIF
      at inference.
    - OSDA → **14-dim descriptor vector** (`syn_variables.osda_cols`:
      asphericity, 2 axes, formal_charge, free SASA, mol_weight, NPR1/2,
      rotatable bonds, PMI1/2/3, spherocity, volume) — RDKit **3D** descriptors
      from SMILES (need conformer gen; precomputed CSVs exist for known OSDAs).
      (`osda_feat_dims=14`.)
  - **Output = 12-dim continuous vector** (`seq_length=12`, quantile-transformed;
    inverse via `dataset.qts`): **10 gel ratios** `Si/Al, Al/P, Si/Ge, Si/B,
    Na/T, K/T, OH/T, F/T, H2O/T, sda1/T` + **2 conditions** `cryst_temp,
    cryst_time`. (`dataset.ratio_names + dataset.cond_names`.) This IS the
    `ConditionSpec` — exact, no longer a placeholder.
  - **Sample API:** `model.sample(zeo=<n×143 tensor>, osda=<n×14 tensor>,
    cond_scale=0.75, batch_size=n) -> n×1×12` in quantile space; inputs scaled by
    pickled `data/scalers/{zeo,osda}_feat_scaler.pkl`; outputs inverse-quantile-
    transformed per column via `data/ZeoSynGen_dataset.pkl` `.qts`.
  - **Loader:** `eval.load_model('diff','run1','system') -> (model, configs)`;
    builds `Unet1D`+`GaussianDiffusion1D`, loads
    `runs/diff/system/run1/best_model.pt`, `.to(device).eval()`. Override
    `configs['device']`→`'cpu'` and `torch.load(map_location='cpu')` for CPU.
  - **All artifacts are IN THE GITHUB REPO** (weights `best_model.pt`, dataset
    pkl, descriptor CSVs, scalers) — **figshare not required.** Deps: torch,
    einops, scikit-learn, pandas, numpy (+ rdkit only for *new* OSDA descriptors).
  - Emphatically **captures multi-modality** (one framework → many valid recipes)
    — the 1000-sample distribution is exactly the diversity MCTS widens over.
- **What it does NOT do (why MCTS is complementary, not redundant):**
  - It does **not choose the OSDA** — it takes it as input. Selecting/designing
    the OSDA is the open discovery decision → **MCTS's job.**
  - It outputs **no feasibility/synthesizability score.** The paper validates
    recipes *post hoc* via **DFT OSDA–framework binding energies** and
    experiment. So DiffSyn *raises* the validation bar; it does not answer it →
    we need a **FeasibilityOracle** seam.
  - It is **zeolite-specific**: its conditioning (framework topology, OSDA) and
    outputs (gel ratios) have **no counterpart in the oxide corpus**, so the
    pretrained model cannot be pointed at oxides. The *method* transfers; the
    *weights* only work for zeolites. This is why Strategy B is a domain pivot.

## 3. Division of labor (the crux of the contribution)

```
MCTS  (outer discovery search)                DiffSyn  (inner generative model, FROZEN)
──────────────────────────────                ────────────────────────────────────────
Given a target framework, searches the        Given (framework, chosen OSDA[, regime]),
DISCRETE / structured decisions:              samples a DISTRIBUTION of realistic
  • which OSDA (from a candidate pool;          continuous conditions (T, time, pH, gel
    later: generated molecule)                   ratios). Also exposes a realism signal
  • mineralizer route (OH- vs F-)                (log-likelihood / on-manifold score).
  • heteroatom / Si-Al regime
At each leaf, queries DiffSyn for conditions
(progressive widening over samples), scores
multi-objectively, backpropagates (PUCT),
returns a DIVERSE PORTFOLIO of novel+feasible
recipes.
                     │
                     ▼
        FeasibilityOracle (grounding)
        DFT OSDA–framework binding energy
        (paper's signal) / surrogate / stub
```

**Why MCTS and not just "enumerate OSDAs and score":** the decision space
(OSDA × mineralizer × heteroatom/Si-Al regime × conditions) is combinatorially
large, the objective is **multi-objective** (feasibility × novelty × realism ×
target-match), we want a **diverse portfolio** rather than a single argmax, and
the oracle call (DFT binding energy) is **expensive**. PUCT allocates the
oracle/generator budget toward promising, novel, diverse regions instead of
exhaustive enumeration. Honesty note: with a *fixed small OSDA pool* the tree is
wide-and-shallow (≈ a multi-objective bandit) — still a legitimate use, but the
*depth* justification for MCTS strengthens greatly once **OSDA design** (sequential
molecular construction) replaces OSDA selection. That is the Phase 3 lever
(§7) that makes the search genuinely deep; Phase 1 keeps OSDA-from-pool.

## 4. Reward (per candidate recipe = OSDA + regime + condition vector)

`R = w_f·feasibility + w_n·novelty + w_r·realism + w_t·target_match`  (all in [0,1])

- **feasibility** — `FeasibilityOracle.score(framework, osda, conditions)`; real
  signal = DFT/surrogate OSDA–framework binding energy mapped to [0,1]. The
  physical gate that keeps novelty from becoming hallucination.
- **novelty** — distance from the nearest **ZeoSyn** recipe for this framework
  (`ZeoliteNoveltyIndex`). The discovery driver; **inverts** the recall
  objective. Novelty is only rewarded *because* it is multiplied/added against
  feasibility+realism.
- **realism** — DiffSyn's own likelihood / on-manifold score for the sampled
  conditions. Keeps proposed conditions physically plausible; prevents the search
  from exploiting oracle blind spots with absurd conditions.
- **target_match** — does the (OSDA, conditions) plausibly template the *target*
  framework (vs some other)? Partly the oracle again; in ZeoSyn it can be seeded
  by empirical OSDA↔framework co-occurrence.

Weights live in `SearchConfig`; defaults documented there. Mean-centering /
neutral-0.5 discipline follows the physics-provider precedent so an unavailable
signal never dominates.

## 5. Engineering: modules, seams, data contracts

New self-contained subpackage `src/synthesis_planner/discovery/`:

| module | responsibility | real vs stub |
|---|---|---|
| `zeolite_schema.py` | `ZeoliteTarget`, `OSDA`, `ConditionSpec` (variable schema), `ConditionVector`, `ZeoliteRecipe`, `SynthesisChoice` | real (pure dataclasses) |
| `condition_generator.py` | `ConditionGenerator` **Protocol** + `DiffSynConditionGenerator` (lazy torch/zeosyn_gen adapter) + `StubConditionGenerator` (offline, deterministic, contract-faithful) | Protocol+stub real; DiffSyn adapter written, runs only where weights+torch exist |
| `oracle.py` | `FeasibilityOracle` Protocol + `BindingEnergyOracle` (DFT/surrogate path, documented) + `StubOracle` (co-occurrence/heuristic) | Protocol+stub real; binding-energy path documented seam |
| `novelty.py` | `ZeoliteNoveltyIndex` — distance-to-corpus over recipes | real; corpus injected (ZeoSyn when available, small set for tests) |
| `search.py` | `DiffSynDiscoverySearch` — self-contained PUCT over discrete choices + progressive widening over sampled conditions; multi-objective reward; diverse portfolio | real, fully testable on stubs |

**Key interface contracts** (code against these; DiffSyn plugs in behind them):

```python
class ConditionGenerator(Protocol):
    def sample(self, target: ZeoliteTarget, osda: OSDA, n: int, *, seed: int | None = ...) \
        -> list[ConditionVector]: ...
    def realism(self, target: ZeoliteTarget, osda: OSDA, conditions: ConditionVector) \
        -> float: ...   # [0,1]; DiffSyn: normalized model log-likelihood / density

class FeasibilityOracle(Protocol):
    def score(self, target: ZeoliteTarget, osda: OSDA, conditions: ConditionVector) \
        -> float: ...   # [0,1]; real: DFT/surrogate binding energy mapped to [0,1]
    def binding_energy(self, target: ZeoliteTarget, osda: OSDA) -> float | None: ...
```

`ConditionSpec` names the ordered continuous variables and their (lo, hi) ranges
for normalization; it is the single place to reconcile with DiffSyn's
`data/syn_variables.py`. `ConditionVector` is a `dict[name -> value]` + the spec.

## 6. Infra / how the frozen weights drop in (no retraining)

On a machine **with a GPU and network**:
1. `pip install '.[diffsyn]'` — adds `torch`, `torch-geometric`, `rdkit`, and the
   `zeosyn_gen` package (from GitHub) as an **optional extra**, lazily imported
   (same discipline as `.[mp]` / `.[judge]`; core stays numpy+pandas).
2. Download **ZeoSyn** (figshare DOI above) → provides the novelty corpus + OSDA
   pool + co-occurrence priors. Download the **DiffSyn checkpoint** from the repo.
3. Config (`config.py`): `{"diffsyn": {"checkpoint": "/path/ckpt.pt", "device":
   "cuda", "syn_variables": "/path/syn_variables.py"}}`.
4. `gen = DiffSynConditionGenerator.from_config(cfg)` — everything downstream is
   unchanged because `search.py` only ever sees the `ConditionGenerator` Protocol.

**Environment reality (re-checked 2026-09-14, correcting an earlier wrong
claim):** this LLNL login node **does** have `uv` (0.11.27) and **PyPI + GitHub
are reachable** (so torch/PyG/rdkit are installable and `zeosyn_gen` is
clonable), but there is **no GPU** and network throughput is **slow** (a blobless
clone of `zeosyn_gen` exceeds a 2-min wall — run downloads in the background).
figshare (likely host of ZeoSyn + the checkpoint) timed out once; needs
confirming. Implication: **CPU inference of the frozen DiffSyn model is feasible
here** (it is a small denoiser over a low-dim condition vector — no training),
so real Phase-2 wiring is *possible in this env*, not blocked. Phase 1 is still
built against `StubConditionGenerator`/`StubOracle` (fast, deterministic,
contract-faithful) because the Protocol boundary is the correct design regardless
— it keeps torch an **optional** dep and makes the coupling testable in
milliseconds. **De-risking pilot (in progress):** clone `zeosyn_gen`, read
`data/syn_variables.py` (the exact variable schema) and the model class's
sample API, locate the checkpoint (repo vs figshare), stand up a CPU `uv` env,
run pretrained inference for one framework, then reconcile `ConditionSpec` and
implement `DiffSynConditionGenerator` against the real API.

## 7. Phased roadmap

- **Phase 1 (this work): the integration architecture, stub-validated.** Schema,
  the three Protocol seams, stub generator + stub oracle + novelty index, the
  PUCT discovery search with progressive widening + multi-objective reward +
  diverse portfolio, tests on stubs. DiffSyn adapter written but not run here.
- **Phase 2: wire real weights** on a GPU box — implement/verify
  `DiffSynConditionGenerator` against `zeosyn_gen`, load ZeoSyn for the novelty
  corpus + OSDA pool + co-occurrence, reconcile `ConditionSpec` with
  `syn_variables.py`. Swap stubs for real; run discovery on a few frameworks.
- **Phase 3: the depth lever — OSDA *design*.** Replace OSDA-from-pool with a
  sequential molecular construction action space (fragment/SMILES tokens), making
  the tree genuinely deep so MCTS clearly beats enumeration; add a DFT/surrogate
  binding-energy oracle for real feasibility.
- **Phase 4: evaluation for the paper.** Discovery metrics (feasible-novelty
  rate, novel-system coverage), leakage-controlled framework hold-outs, ablations
  (−novelty, −realism, −oracle), and — venue-defining — a handful of
  DFT/experimental confirmations of discovered recipes.

## 8. Open questions / risks (track these)

- Exact `syn_variables.py` schema + variable ranges — unknown until Phase 2;
  `ConditionSpec` is a documented placeholder.
- Does `zeosyn_gen` expose a clean programmatic `sample(framework, osda, n)` and a
  likelihood, or only `predict.py` CLI? May need a thin wrapper around their model
  class. Adapter is written defensively (lazy import + documented call site).
- Licensing: `zeosyn_gen` code license + ZeoSyn figshare license — audit before
  redistributing any derived data (same as the Kononova/Wang audit).
- Feasibility oracle: DFT binding energy is expensive; a learned surrogate may be
  needed for search-scale scoring. Oracle seam supports either.
- OSDA-from-pool makes Phase-1 search shallow; the honest MCTS-vs-enumeration
  claim depends on Phase 3.

---

## STATUS (update this on every session)

- **2026-09-14 — Phase 1 COMPLETE (stub-validated), Phase 2 groundwork done.**
  - [x] `discovery/zeolite_schema.py` — exact 12-var `ConditionSpec`, domain types.
  - [x] `discovery/condition_generator.py` — `ConditionGenerator` Protocol +
        `StubConditionGenerator` (offline, contract-faithful) +
        `DiffSynConditionGenerator` (real CPU adapter, lazy torch, replicates the
        repo's exact 'diff' model build to load `model.pt` with map_location=cpu).
  - [x] `discovery/oracle.py` — `FeasibilityOracle` Protocol + `BindingEnergyOracle`
        (DFT/surrogate seam) + `CooccurrenceOracle` (ZeoSyn prior stub).
  - [x] `discovery/novelty.py` — `ZeoliteNoveltyIndex` (distance-to-corpus).
  - [x] `discovery/search.py` — `DiffSynDiscoverySearch`: PUCT over OSDA/mineralizer,
        progressive widening over generator samples, multi-objective reward
        (feasibility x novelty x realism), diverse portfolio.
  - [x] `tests/test_diffsyn_integration.py` — 15 tests, all green; full suite
        **154 passed / 1 skipped**. Locks contract, coupling, PW, novelty-vs-recall,
        diversity, and that importing the DiffSyn adapter is torch-free.
  - [x] `.[diffsyn]` extra + `synthesis_planner.discovery` registered in pyproject.
  - **Env recheck (important):** `uv` present, PyPI+GitHub reachable, **no GPU**,
    slow net, **home (`/g/g10`) has a tight disk quota** — put venvs/caches on
    lustre. A CPU torch env was built at `/p/lustre2/laubach2/diffsyn-venv`
    (UV_CACHE_DIR/UV_PYTHON_INSTALL_DIR set to `/p/lustre2/laubach2/.uv-*`). The
    `zeosyn_gen` repo is blobless-cloned at
    `<scratchpad>/zeosyn_gen` (weights/data blobs not yet fetched — they are large;
    fetch `runs/diff/system/run1/model.pt`, `data/ZeoSynGen_dataset.pkl`,
    `data/zeolite_descriptors.csv`, `data/scalers/*.pkl`, an OSDA feature CSV).

  ### Phase 2 — NEXT ACTIONS (real DiffSyn inference on CPU)
  **CONCRETE FACTS discovered 2026-09-14 (read before retrying):**
  - The diff **`model.pt` is NOT in git** — download from Dropbox (per repo README):
    `wget -O runs/diff/system/run1/model.pt "https://www.dropbox.com/scl/fi/vmf5ag87vszlikmlsnlg4/model.pt?rlkey=9p1d2ht0qxr32of0xizsmqxat&st=obgh0a2n&dl=1"`
    (repo data DOI is Zenodo 10.5281/zenodo.17645370, not figshare.)
  - Importing `models/diffusion.py` pulls a chain of **vestigial** (plotting/train)
    top-level imports on top of torch/einops. Full set discovered by iterating the
    smoke: `torchvision matplotlib seaborn tqdm scipy pyrolite accelerate
    ema_pytorch` (all pip-installable). Our adapter deliberately does NOT import
    `eval.py` (which also needs `torch_geometric`), sidestepping that fragile dep.
    Turnkey: `uv pip install --python <diffsyn-venv> torchvision matplotlib seaborn
    tqdm scipy pyrolite accelerate ema_pytorch` (torchvision via the cpu index).
  - **sklearn pickle-compat:** the repo pins `scikit-learn==1.1.2`; the
    `data/scalers/*.pkl` + `ZeoSynGen_dataset.pkl` (quantile transformers) were
    pickled with it. Our env has a newer sklearn — if `.transform` /
    `.inverse_transform` errors or warns, pin `scikit-learn==1.1.2` (may also need
    `numpy==1.26 pandas==1.4.3`) OR bypass: run the model to get the raw 12-dim
    output and skip the inverse quantile-transform for a "model-runs" smoke.
  - `git checkout` **aborts the whole command on one bad pathspec** — that's why the
    first blob fetch pulled nothing; list only paths that exist in git.
  - Dataset pickle class: `data/ZeoSynGen_dataset.pkl` unpickles a custom class
    exposing `.ratio_names/.cond_names/.qts`; its defining module must be importable
    at load. Location still TBD (grep the repo for `ratio_names`/`class .*Dataset`);
    if it lives in a notebook, replicate a minimal class or extract `.qts` directly.
  1. In `<scratchpad>/zeosyn_gen` (`git config gc.auto 0` first) fetch the in-git
     blobs: `git checkout HEAD -- data/ZeoSynGen_dataset.pkl data/zeolite_descriptors.csv
     data/scalers data/2024-10-02_K222_and_CHA_OSDA_features.csv data/syn_variables.py
     data/iza_codes.py models/` then wget `model.pt` (URL above). (Clone is on lustre
     scratch — good, avoids the home quota.)
  2. Sanity-check versions: the pickled scalers/`ZeoSynGen_dataset.pkl` were made
     with some sklearn version; if unpickling errors, match sklearn in the
     `diffsyn-venv`. `model.pt` is a plain state_dict (map_location=cpu handles device).
  3. Smoke: `from synthesis_planner.discovery import DiffSynConditionGenerator`,
     construct with `repo_root=<clone>`, `.sample(ZeoliteTarget('CHA'), OSDA(<a
     SMILES present in the OSDA CSV>), n=16)`; assert 16 vectors, 12 vars,
     physical ranges sane. Then plug it into `DiffSynDiscoverySearch` in place of
     the stub — nothing else changes.
  4. Build the real novelty corpus + OSDA pool + `CooccurrenceOracle` table from
     `ZeoSynGen_dataset.pkl` / `data/ZEOSYN.xlsx`.
  - **Risk watch:** OSDA descriptors for OSDAs NOT in the CSV need the repo's
    RDKit 3D featurizer (deferred to Phase 3 OSDA-design). `.realism()` currently
    draws a 128-sample reference per call — cache it per (target,osda) before
    running large searches with the real (slower) generator.
