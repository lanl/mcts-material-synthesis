"""Shared pytest fixtures for the mcts_crystal test suite."""

from pathlib import Path

import pandas as pd
import pytest
from ase.io import read

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def u_w_pb_atoms():
    """The default starting structure (Pb6UW6, space group 191) - no heavy deps needed to read a CIF."""
    return read(str(REPO_ROOT / "examples" / "mat_Pb6U1W6_sg191.cif"))


@pytest.fixture
def doscar_rewards_csv(tmp_path):
    """A small synthetic doscar_rewards.csv for DoscarRewardLookup tests."""
    csv_path = tmp_path / "doscar_rewards.csv"
    df = pd.DataFrame({
        "compound_name": ["U-Pb-W", "Ce-Si-Ti"],
        "reward_normalized": [0.42, 1.23],
    })
    df.to_csv(csv_path, index=False)
    return csv_path
