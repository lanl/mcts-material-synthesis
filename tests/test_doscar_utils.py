"""Tests for mcts_crystal.doscar_utils.DoscarRewardLookup."""

import pytest

from mcts_crystal.doscar_utils import DoscarRewardLookup


class TestFormulaConversion:
    def test_converts_mcts_formula_to_doscar_format(self):
        lookup = DoscarRewardLookup(csv_file="/nonexistent.csv")
        assert lookup.convert_formula_to_doscar_format("Ti6Si6Ce") == "Ce-Si-Ti"

    def test_order_independent(self):
        lookup = DoscarRewardLookup(csv_file="/nonexistent.csv")
        assert lookup.convert_formula_to_doscar_format("U1Pb6W6") == "U-Pb-W"

    def test_returns_none_for_wrong_element_count(self):
        lookup = DoscarRewardLookup(csv_file="/nonexistent.csv")
        assert lookup.convert_formula_to_doscar_format("Ti6Si6") is None


class TestGetReward:
    def test_loads_and_finds_known_compound(self, doscar_rewards_csv):
        lookup = DoscarRewardLookup(csv_file=str(doscar_rewards_csv))
        assert lookup.get_reward("Pb6UW6") == pytest.approx(0.42)
        assert lookup.get_reward("Ti6Si6Ce") == pytest.approx(1.23)

    def test_unknown_compound_returns_zero(self, doscar_rewards_csv):
        lookup = DoscarRewardLookup(csv_file=str(doscar_rewards_csv))
        assert lookup.get_reward("Ti6Si6Nd") == 0.0

    def test_missing_csv_file_degrades_to_zero_rewards(self):
        lookup = DoscarRewardLookup(csv_file="/definitely/not/a/real/path.csv")
        assert lookup.rewards_dict == {}
        assert lookup.get_reward("Pb6UW6") == 0.0
