"""Tests for mcts_crystal.mcts.MCTS, using stub energy/doscar calculators (no real MACE/MP calls)."""

import pytest

from mcts_crystal.mcts import MCTS
from mcts_crystal.node import MCTSTreeNode, ehull_reward


class StubEnergyCalculator:
    """Deterministic stand-in for MaceEnergyCalculator - no MACE/Materials Project needed."""

    def calculate_energies(self, atoms):
        # Stable-ish, deterministic values so reward calculations are predictable
        return -0.5, 0.02


class StubDoscarLookup:
    def __init__(self, reward=0.3):
        self.reward = reward

    def get_reward(self, formula):
        return self.reward


@pytest.fixture
def root_node(u_w_pb_atoms):
    return MCTSTreeNode(u_w_pb_atoms, f_block_mode='u_only', termination_limit=10)


class TestUCBAndSelection:
    def test_root_ucb_is_infinite_before_any_visits(self, root_node):
        assert root_node.get_ucb() == float('inf')

    def test_select_node_returns_root_when_not_expanded(self, root_node):
        mcts = MCTS(root_node)
        chain = mcts.select_node()
        assert chain == [root_node]


class TestExpansionSimulation:
    def test_ehull_reward_matches_node_formula(self, root_node):
        mcts = MCTS(root_node)
        reward, _ = mcts.expansion_simulation(
            rollout_depth=0, n_rollout=1,
            energy_calculator=StubEnergyCalculator(),
            rollout_method='ehull',
        )
        assert reward == pytest.approx(ehull_reward(0.02))

    def test_ehull_rdos_combines_both_terms(self, root_node):
        mcts = MCTS(root_node)
        reward, _ = mcts.expansion_simulation(
            rollout_depth=0, n_rollout=1,
            energy_calculator=StubEnergyCalculator(),
            rollout_method='ehull_rdos',
            beta=1.0, gamma=2.5,
            doscar_lookup=StubDoscarLookup(reward=0.3),
        )
        assert reward == pytest.approx(1.0 * ehull_reward(0.02) + 2.5 * 0.3)

    def test_rdos_only_ignores_energy_calculator(self, root_node):
        mcts = MCTS(root_node)
        reward, _ = mcts.expansion_simulation(
            rollout_depth=0, n_rollout=1,
            energy_calculator=None,
            rollout_method='rdos',
            doscar_lookup=StubDoscarLookup(reward=0.9),
        )
        assert reward == pytest.approx(0.9)

    def test_unknown_rollout_method_raises(self, root_node):
        mcts = MCTS(root_node)
        with pytest.raises(ValueError):
            mcts.expansion_simulation(
                rollout_depth=0, n_rollout=1,
                energy_calculator=StubEnergyCalculator(),
                rollout_method='not_a_real_method',
            )


class TestRunIntegration:
    @pytest.mark.parametrize("rollout_method", ["ehull", "ehull_rdos", "rdos"])
    def test_run_completes_for_each_rollout_method(self, root_node, rollout_method):
        mcts = MCTS(root_node)
        results = mcts.run(
            n_iterations=5,
            energy_calculator=StubEnergyCalculator(),
            rollout_depth=0,
            n_rollout=1,
            rollout_method=rollout_method,
            beta=1.0,
            gamma=2.5,
            doscar_lookup=StubDoscarLookup(),
        )
        assert results['iterations_completed'] >= 1
        assert len(results['stat_dict']) >= 1
