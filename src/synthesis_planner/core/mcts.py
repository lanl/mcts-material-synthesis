"""A small PUCT-style MCTS engine for synthesis routes."""

from __future__ import annotations

import math
import random

from .grammar import apply_action, expand_state, rollout_completion
from .schema import Action, EvaluationConfig, PlanningState
from .scoring import evaluate_state


class TreeNode:
    def __init__(self, state: PlanningState, prior: float = 1.0, parent: "TreeNode | None" = None):
        self.state = state
        self.prior = prior
        self.parent = parent
        self.children: list[TreeNode] = []
        self.visit_count = 0
        self.total_value = 0.0
        # Best (max) rollout value seen through this node - used only for
        # reporting/diagnostics; selection is driven by the mean q_value.
        self.best_value = float("-inf")
        self.expanded = False
        self.terminal_routes = []

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


class MonteCarloTreeSearch:
    """PUCT tree search over synthesis-route completions.

    ``value_aggregation`` controls what a simulation backs up:
      - ``"mean"`` (default): the expected value of the random rollouts, so a
        node's ``q_value`` is a genuine value estimate and PUCT concentrates
        search on branches whose completions are consistently good.
      - ``"max"``: the best rollout value (optimistic; the previous behavior).
    Either way the single best-scoring rollout route is retained for the
    portfolio, so switching aggregation never lowers final route quality.
    """

    def __init__(
        self,
        exploration_constant: float = 1.4,
        rollout_count: int = 8,
        seed: int | None = None,
        evaluation_config: EvaluationConfig | None = None,
        mp_client=None,
        value_aggregation: str = "mean",
        templates=None,
    ):
        self.exploration_constant = exploration_constant
        self.rollout_count = rollout_count
        self.rng = random.Random(seed)
        self.evaluation_config = evaluation_config or EvaluationConfig()
        self.mp_client = mp_client
        # Optional mined TemplateLibrary (WS-B), duck-typed; widens the heating
        # action space via per-family condition schedules when present.
        self.templates = templates
        if value_aggregation not in {"mean", "max"}:
            raise ValueError(f"value_aggregation must be 'mean' or 'max', got {value_aggregation!r}")
        self.value_aggregation = value_aggregation

    def run(self, root_state: PlanningState, analogs, candidate_precursor_sets, iterations: int):
        root = TreeNode(root_state)

        for _ in range(iterations):
            node = self._select(root)
            if not node.state.is_terminal:
                self._expand(node, analogs, candidate_precursor_sets)
                if node.children:
                    node = self._select_child_for_rollout(node)

            value, route = self._simulate(node.state, analogs, candidate_precursor_sets)
            self._backup(node, value, route)

        return root

    def _select(self, node: TreeNode) -> TreeNode:
        current = node
        while current.children:
            current = max(current.children, key=lambda child: self._puct_score(current, child))
        return current

    def _select_child_for_rollout(self, node: TreeNode) -> TreeNode:
        """Pick the freshly expanded child to roll out from, weighted by prior.

        Prior-weighted (rather than uniform) sampling means the first value
        estimates flow to the actions the grammar/retrieval consider most
        promising, giving PUCT better-seeded statistics early on.
        """
        priors = [max(child.prior, 0.0) for child in node.children]
        total = sum(priors)
        if total <= 0.0:
            return self.rng.choice(node.children)
        threshold = self.rng.random() * total
        cumulative = 0.0
        for child, prior in zip(node.children, priors):
            cumulative += prior
            if cumulative >= threshold:
                return child
        return node.children[-1]

    def _expand(self, node: TreeNode, analogs, candidate_precursor_sets) -> None:
        if node.expanded or node.state.is_terminal:
            return
        actions = expand_state(node.state, analogs, candidate_precursor_sets, self.templates)
        # Normalize sibling priors to sum to 1 (AlphaZero-style) so the PUCT
        # exploration term is comparably scaled regardless of how many actions
        # the grammar emits or how the raw priors are magnituded.
        total_prior = sum(max(action.prior, 0.0) for action in actions)
        for action in actions:
            child_state = apply_action(node.state, action, analogs)
            normalized_prior = (max(action.prior, 0.0) / total_prior) if total_prior > 0.0 else (1.0 / len(actions))
            node.children.append(TreeNode(child_state, prior=normalized_prior, parent=node))
        node.expanded = True

    def _simulate(self, state: PlanningState, analogs, candidate_precursor_sets):
        best_route = None
        best_value = float("-inf")
        value_sum = 0.0
        count = 0
        for _ in range(self.rollout_count):
            terminal_state = rollout_completion(state, analogs, candidate_precursor_sets, self.rng, self.templates)
            route = evaluate_state(terminal_state, analogs, self.evaluation_config, mp_client=self.mp_client)
            value_sum += route.mcts_value
            count += 1
            if route.mcts_value > best_value:
                best_value = route.mcts_value
                best_route = route
        if count == 0:
            return 0.0, None
        backup_value = best_value if self.value_aggregation == "max" else value_sum / count
        return backup_value, best_route

    def _backup(self, node: TreeNode, value: float, route) -> None:
        current = node
        while current is not None:
            current.visit_count += 1
            current.total_value += value
            if value > current.best_value:
                current.best_value = value
            if route is not None:
                current.terminal_routes.append(route)
            current = current.parent

    def _puct_score(self, parent: TreeNode, child: TreeNode) -> float:
        exploration = self.exploration_constant * child.prior * math.sqrt(parent.visit_count + 1) / (1 + child.visit_count)
        return child.q_value + exploration
