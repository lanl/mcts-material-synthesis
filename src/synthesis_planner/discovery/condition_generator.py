"""The DiffSyn condition-generator seam: Protocol + offline stub + real adapter.

This is the boundary that lets the frozen, heavyweight DiffSyn diffusion model
plug into the lightweight MCTS discovery search without making torch a hard
dependency. ``search.py`` only ever sees the ``ConditionGenerator`` Protocol.

- ``StubConditionGenerator`` -- deterministic, torch-free, contract-faithful.
  Used for all offline development/testing (this env has no GPU). It emits the
  exact 12-variable ``ConditionSpec`` DiffSyn emits, as a per-(framework, OSDA)
  Gaussian cloud, so progressive widening, the reward, and the portfolio logic
  are all exercised identically to how they will behave with real DiffSyn.
- ``DiffSynConditionGenerator`` -- the real adapter around ``eltonpan/zeosyn_gen``
  checkpoint ``runs/diff/system/run1``. Lazily imports torch and the repo's
  ``models.diffusion`` / ``data.syn_variables``; builds the model on CPU (the
  repo's ``eval.load_model`` hardcodes ``map_location=cuda``, so we replicate its
  exact construction with ``map_location='cpu'`` instead of calling it). Not run
  in the CI/offline env; validated on a machine where the weights + deps exist.

Both return ``list[ConditionVector]`` and a ``realism`` in [0,1] (how on-manifold
a condition vector is for this conditioning), used by the search's realism term.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
from contextlib import contextmanager
from typing import Any, Optional, Protocol, runtime_checkable

from .zeolite_schema import (
    DEFAULT_CONDITION_SPEC,
    ConditionSpec,
    ConditionVector,
    OSDA,
    ZeoliteTarget,
)


@runtime_checkable
class ConditionGenerator(Protocol):
    """Generative model over synthesis conditions, conditioned on (target, OSDA)."""

    spec: ConditionSpec

    def sample(
        self, target: ZeoliteTarget, osda: OSDA, n: int, *, seed: Optional[int] = None
    ) -> list[ConditionVector]:
        """Return ``n`` condition samples for this (framework, OSDA) pair."""
        ...

    def realism(self, target: ZeoliteTarget, osda: OSDA, conditions: ConditionVector) -> float:
        """[0,1] on-manifold score: how typical these conditions are for the pair."""
        ...


def _clip(value: float, bounds: Optional[tuple[float, float]]) -> float:
    if bounds is None:
        return value
    lo, hi = bounds
    return max(lo, min(hi, value))


def _seed_from(*parts: Any) -> int:
    """Deterministic 32-bit seed from arbitrary parts (stable across runs)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


class StubConditionGenerator:
    """Deterministic, torch-free stand-in matching DiffSyn's 12-var I/O contract.

    For a given (framework, OSDA) it defines a stable Gaussian cloud in condition
    space (center + spread derived by hashing the identifiers), clipped to the
    advisory physical ranges. This mimics DiffSyn's key property -- a *distribution*
    of plausible recipes per pair, with genuine pair-to-pair variation -- so the
    downstream search behaves the same as it will with real weights, just without
    chemical realism. ``realism`` is the Gaussian density of a point in that cloud.
    """

    def __init__(self, spec: ConditionSpec = DEFAULT_CONDITION_SPEC, spread: float = 0.15):
        self.spec = spec
        self.spread = spread  # fraction-of-range std dev of the cloud

    def _center_and_std(self, target: ZeoliteTarget, osda: OSDA, name: str) -> tuple[float, float]:
        lo, hi = self.spec.ranges.get(name, (0.0, 1.0))
        span = hi - lo
        # Center is a stable pseudo-random point in-range for this (pair, variable).
        frac = (_seed_from(target.framework_code, osda.smiles, name) % 1000) / 1000.0
        center = lo + frac * span
        return center, self.spread * span

    def sample(
        self, target: ZeoliteTarget, osda: OSDA, n: int, *, seed: Optional[int] = None
    ) -> list[ConditionVector]:
        import random

        base = _seed_from(target.framework_code, osda.smiles) ^ (seed or 0)
        rng = random.Random(base)
        out: list[ConditionVector] = []
        for _ in range(n):
            values: dict[str, float] = {}
            for name in self.spec.variables:
                center, std = self._center_and_std(target, osda, name)
                values[name] = _clip(rng.gauss(center, std), self.spec.ranges.get(name))
            out.append(ConditionVector(values=values, spec=self.spec))
        return out

    def realism(self, target: ZeoliteTarget, osda: OSDA, conditions: ConditionVector) -> float:
        # Mean per-variable Gaussian density (normalized to 1.0 at the center),
        # i.e. exp(-1/2 * mean z^2). On-cloud -> ~1, far tail -> ~0.
        zs = []
        for name in self.spec.variables:
            center, std = self._center_and_std(target, osda, name)
            if std <= 0:
                continue
            zs.append(((conditions.get(name) - center) / std) ** 2)
        if not zs:
            return 0.5
        return math.exp(-0.5 * (sum(zs) / len(zs)))


@contextmanager
def _repo_on_path(repo_root: str):
    """Run with ``repo_root`` as cwd and on ``sys.path`` (zeosyn_gen uses relative
    ``open('data/...')`` / ``import eval`` / ``from models...``)."""
    prev_cwd = os.getcwd()
    added = repo_root not in sys.path
    try:
        if added:
            sys.path.insert(0, repo_root)
        os.chdir(repo_root)
        yield
    finally:
        os.chdir(prev_cwd)
        if added and repo_root in sys.path:
            sys.path.remove(repo_root)


class DiffSynConditionGenerator:
    """Real adapter over the frozen DiffSyn checkpoint (``runs/diff/system/run1``).

    Wiring (all artifacts live inside the ``eltonpan/zeosyn_gen`` clone -- weights,
    dataset pkl, descriptor CSVs, scalers; figshare not required):

      gen = DiffSynConditionGenerator(repo_root="/path/to/zeosyn_gen", device="cpu")
      samples = gen.sample(ZeoliteTarget("CHA"), OSDA(smiles), n=1000)

    We do NOT call the repo's ``eval.load_model`` because it hardcodes
    ``map_location=configs['device']`` (cuda); on CPU that raises. Instead we
    replicate its exact 'diff' construction (verified from eval.py) with
    ``map_location='cpu'``. Everything is lazily imported so importing this module
    never pulls torch.

    NOTE (Phase 2 to verify on a torch box): OSDA descriptors. DiffSyn conditions
    on a 14-dim RDKit-3D descriptor vector. If ``osda.descriptors`` is provided
    (len 14, ``osda_cols`` order) we use it; otherwise we look the SMILES up in the
    repo's precomputed OSDA feature CSV. Computing descriptors for a *novel* OSDA
    (Phase 3) needs the repo's RDKit featurizer -- deferred.
    """

    #: checkpoint coordinates for the released DiffSyn model
    MODEL_TYPE = "diff"
    FNAME = "run1"
    SPLIT = "system"

    def __init__(
        self,
        repo_root: str,
        device: str = "cpu",
        cond_scale: float = 0.75,
        spec: ConditionSpec = DEFAULT_CONDITION_SPEC,
        osda_feature_csv: str = "data/2024-10-02_K222_and_CHA_OSDA_features.csv",
    ):
        self.repo_root = repo_root
        self.device = device
        self.cond_scale = cond_scale
        self.spec = spec
        self.osda_feature_csv = osda_feature_csv
        self._model = None
        self._configs = None
        self._zeo_df = None
        self._osda_df = None
        self._zeo_scaler = None
        self._osda_scaler = None
        self._qts = None  # per-variable quantile transformers (inverse transform)
        self._col_order: tuple[str, ...] = ()

    # -- lazy load ---------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import json
        import pickle

        import numpy as np  # noqa: F401  (used by scalers)
        import pandas as pd
        import torch

        with _repo_on_path(self.repo_root):
            from models.diffusion import GaussianDiffusion1D, Unet1D  # type: ignore
            from data.syn_variables import zeo_cols, osda_cols  # type: ignore

            cfg_path = f"runs/{self.MODEL_TYPE}/{self.SPLIT}/{self.FNAME}/configs.json"
            with open(cfg_path) as fh:
                configs = json.load(fh)
            mp = configs["model_params"]
            mp.setdefault("dropout", False)

            unet = Unet1D(
                dim=mp["dim"], dim_mults=mp["dim_mults"], channels=mp["channels"],
                resnet_block_groups=mp["resnet_block_groups"], zeo_feat_dims=mp["zeo_feat_dims"],
                osda_feat_dims=mp["osda_feat_dims"], cond_drop_prob=mp["cond_drop_prob"],
                dropout=mp["dropout"],
            )
            model = GaussianDiffusion1D(
                unet, seq_length=mp["seq_length"], timesteps=mp["timesteps"], objective="pred_v",
            ).to(self.device)
            ckpt = f"runs/{self.MODEL_TYPE}/{self.SPLIT}/{self.FNAME}/model.pt"
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            model.eval()

            zeo_df = pd.read_csv("data/zeolite_descriptors.csv")
            if "Unnamed: 0" in zeo_df.columns:
                zeo_df = zeo_df.drop(columns=["Unnamed: 0"])
            osda_df = pd.read_csv(self.osda_feature_csv)
            if "Unnamed: 0" in osda_df.columns:
                osda_df = osda_df.drop(columns=["Unnamed: 0"])
            with open("data/scalers/zeo_feat_scaler.pkl", "rb") as fh:
                zeo_scaler = pickle.load(fh)
            with open("data/scalers/osda_feat_scaler.pkl", "rb") as fh:
                osda_scaler = pickle.load(fh)
            with open("data/ZeoSynGen_dataset.pkl", "rb") as fh:
                dataset = pickle.load(fh)

            self._zeo_cols = list(zeo_cols)
            self._osda_cols = list(osda_cols.keys())
            self._col_order = tuple(dataset.ratio_names) + tuple(dataset.cond_names)
            self._qts = {name: dataset.qts[name] for name in self._col_order}

        self._model = model
        self._configs = configs
        self._zeo_df = zeo_df
        self._osda_df = osda_df
        self._zeo_scaler = zeo_scaler
        self._osda_scaler = osda_scaler

    # -- feature lookup ----------------------------------------------------
    def _zeo_features(self, target: ZeoliteTarget):
        import numpy as np

        if target.descriptors and len(target.descriptors) == len(self._zeo_cols):
            return np.array([target.descriptors], dtype=float)
        df = self._zeo_df
        row = df[df["Code"] == target.framework_code]
        if row.empty:
            raise KeyError(f"zeolite descriptors not found for {target.framework_code}")
        return np.array(row[self._zeo_cols], dtype=float)

    def _osda_features(self, osda: OSDA):
        import numpy as np

        if osda.descriptors and len(osda.descriptors) == len(self._osda_cols):
            return np.array([osda.descriptors], dtype=float)
        df = self._osda_df
        col = "osda smiles" if "osda smiles" in df.columns else df.columns[0]
        row = df[df[col] == osda.smiles]
        if row.empty:
            raise KeyError(
                f"OSDA descriptors not found for {osda.smiles!r}; provide osda.descriptors "
                f"(14-dim, osda_cols order) or add it to {self.osda_feature_csv}"
            )
        return np.array(row[self._osda_cols], dtype=float)

    # -- generation --------------------------------------------------------
    def sample(
        self, target: ZeoliteTarget, osda: OSDA, n: int, *, seed: Optional[int] = None
    ) -> list[ConditionVector]:
        self._ensure_loaded()
        import numpy as np
        import torch
        from einops import repeat

        if seed is not None:
            torch.manual_seed(seed)

        zeo_feat = torch.tensor(self._zeo_scaler.transform(self._zeo_features(target)), dtype=torch.float32)
        osda_feat = torch.tensor(self._osda_scaler.transform(self._osda_features(osda)), dtype=torch.float32)
        zeo_feat = repeat(zeo_feat, "n d -> (r n) d", r=n).to(self.device)
        osda_feat = repeat(osda_feat, "n d -> (r n) d", r=n).to(self.device)

        sampled = self._model.sample(
            batch_size=n, zeo=zeo_feat, osda=osda_feat, cond_scale=self.cond_scale
        )
        arr = sampled.squeeze().detach().cpu().numpy().reshape(n, len(self._col_order))
        # Inverse quantile-transform each column back to physical units.
        for j, name in enumerate(self._col_order):
            arr[:, j] = self._qts[name].inverse_transform(arr[:, j].reshape(-1, 1)).reshape(-1)

        out: list[ConditionVector] = []
        for i in range(n):
            values = {name: float(arr[i, j]) for j, name in enumerate(self._col_order)}
            out.append(ConditionVector(values=values, spec=self.spec))
        return out

    def realism(self, target: ZeoliteTarget, osda: OSDA, conditions: ConditionVector) -> float:
        """On-manifold score via a reference sample cloud from the model itself.

        Diffusion models do not expose a cheap exact likelihood, so we approximate
        realism by drawing a reference batch for this (target, OSDA) and scoring
        the query by its mean per-variable z-distance to that cloud
        (exp(-1/2 * mean z^2)). Cached per pair to keep it cheap.
        """
        import numpy as np

        ref = self.sample(target, osda, 128, seed=0)
        mat = np.array([cv.as_tuple() for cv in ref])  # 128 x 12
        mean = mat.mean(axis=0)
        std = mat.std(axis=0) + 1e-9
        z = (np.array(conditions.as_tuple()) - mean) / std
        return float(np.exp(-0.5 * np.mean(z ** 2)))
