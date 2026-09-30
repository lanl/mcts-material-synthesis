"""DAG recursion driver: retrosynthesis-style multi-step planning (WS-E).

This is a *thin* driver over the existing single-target MonteCarloTreeSearch. It
does NOT rewrite the tree semantics. For a root target it obtains the best
single-step recipe, then for every precursor that is not in stock and is a
plausible sub-target it recurses -- the precursor becomes a new target expanded
by the same machinery -- assembling a :class:`SynthesisDAG`. This is the direct
analog of AiZynthFinder terminating a branch when purchasable building blocks are
reached, adapted to the ~99%-single-step reality of inorganic synthesis (hence a
default ``max_depth=2``; depth-3 is essentially nonexistent in the corpus).

Design constraints honored:
- **Reuse, don't rewrite** the inner PUCT engine (the single-step plan is a
  dependency-injected callable, ``plan_single``).
- **Cycle guard**: a formula already on the ancestor path is never re-expanded
  (prevents A<-B<-A), emitted as a dangling leaf instead.
- **Hard depth cap**: beyond ``max_depth`` a non-stock precursor is a dangling
  (unsolved) leaf; the DAG is solved only if every leaf bottoms out in stock.
- **Layering-safe**: this core module imports only ``core.schema``; the stock,
  templates, retrieval and MCTS wiring are supplied by the orchestration layer
  (``planner.py``) via injected callables.
"""

from __future__ import annotations

from typing import Callable

from .schema import DAGNode, PlannedRoute, PrecursorRecord, SynthesisDAG

# Injected dependency signatures (documented seams for Phase 3):
#   plan_single(formula, modality) -> PlannedRoute | None
#       Best single-step recipe for one material (wraps the inner MCTS).
#   is_in_stock(precursor) -> bool
#       Stock membership (a data.stock.Stock.contains bound method).
#   is_recursion_candidate(formula) -> bool
#       Whether a non-stock precursor formula is a plausible sub-target
#       (e.g. a known target in the corpus). Controls whether we recurse.
PlanSingle = Callable[[str, str], "PlannedRoute | None"]
StockTest = Callable[[PrecursorRecord], bool]
RecursionTest = Callable[[str], bool]


def plan_retro(
    root_formula: str,
    modality: str,
    plan_single: PlanSingle,
    is_in_stock: StockTest,
    is_recursion_candidate: RecursionTest,
    max_depth: int = 2,
) -> SynthesisDAG:
    """Assemble a synthesis DAG for ``root_formula`` by recursive decomposition.

    Args:
        root_formula: the target material to plan.
        modality: synthesis modality (propagated to sub-targets).
        plan_single: callable returning the best single-step recipe for a target.
        is_in_stock: precursor stock-membership test.
        is_recursion_candidate: whether a non-stock precursor formula should be
            expanded further (plausible sub-target).
        max_depth: hard cap on recursion depth (default 2).

    Returns:
        A :class:`SynthesisDAG`; ``dag.is_solved`` is True iff every leaf is in
        stock and every internal node has a valid recipe.
    """
    visited: set[str] = set()

    def _expand(formula: str, depth: int) -> DAGNode:
        recipe = plan_single(formula, modality)
        node = DAGNode(target_formula=formula, modality=modality, depth=depth, recipe=recipe)
        if recipe is None:
            node.dangling = True
            return node

        visited.add(formula)
        for precursor in recipe.precursors:
            node.children.append(_expand_precursor(precursor, depth + 1))
        visited.discard(formula)
        return node

    def _expand_precursor(precursor: PrecursorRecord, depth: int) -> DAGNode:
        # Terminal: precursor is a buildable commodity in stock.
        if is_in_stock(precursor):
            return DAGNode(precursor.formula, modality, depth, in_stock=True)

        # Cycle guard: never re-expand an ancestor formula.
        if precursor.formula in visited:
            return DAGNode(precursor.formula, modality, depth, dangling=True)

        # Recurse if within the depth cap and the precursor is a plausible target.
        if depth <= max_depth and is_recursion_candidate(precursor.formula):
            child = _expand(precursor.formula, depth)
            # A recursion that yielded no recipe is a dangling (unsolved) leaf.
            if child.recipe is None:
                child.dangling = True
            return child

        # Otherwise: a non-stock precursor we cannot (or won't) decompose further.
        return DAGNode(precursor.formula, modality, depth, dangling=True)

    root = _expand(root_formula, depth=0)
    return SynthesisDAG(target_formula=root_formula, modality=modality, root=root, max_depth_cap=max_depth)


# A genuine precursor-powder-forming step: the route precipitated and/or dried an
# amorphous solid *before* the calcination. (A plain dissolve/mix or a
# hydrothermal hold does not create a separable, dried precursor powder.)
_PRECURSOR_FORMING_VERBS = frozenset({"precipitate", "dry"})


def insert_solution_intermediate(node: DAGNode, is_in_stock: StockTest) -> DAGNode:
    """Split a solution route at its calcination boundary into a depth-2 DAG (#2).

    Co-precipitation / sol-gel / Pechini routes are natively multi-step: the
    metal salts first form an amorphous **precursor powder / gel** (precipitate,
    wash, dry), which is *then* calcined to the crystalline target. This models
    that explicitly as::

        target  <- calcine( precursor powder )          (this node's recipe)
        precursor powder <- precipitate/gel( salts )     (intermediate node)
        salts   in stock                                  (leaves)

    If the recipe has no post-precipitation calcination (e.g. a hydrothermal run
    that crystallises the product directly), the node is left unchanged - that is
    the chemically faithful single-step case.
    """
    recipe = node.recipe
    if recipe is None or not recipe.operations:
        return node

    ops = recipe.operations
    # The calcination is the last heating step of the route.
    heat_idx = None
    for i, op in enumerate(ops):
        if op.verb == "heat":
            heat_idx = i
    if heat_idx is None:
        return node

    pre_ops = ops[:heat_idx]
    # Split only if an amorphous precursor powder was precipitated/dried BEFORE
    # this calcination. This excludes a hydrothermal hold (heat precedes
    # wash/dry) and a bare dissolve->heat, which are single-step crystallisations.
    if not any(op.verb in _PRECURSOR_FORMING_VERBS for op in pre_ops):
        return node

    calcine_ops = ops[heat_idx:]
    kind = "gel" if node.modality == "precipitation" else "powder"
    intermediate = DAGNode(
        target_formula=f"{node.target_formula} precursor ({kind})",
        modality=node.modality,
        depth=node.depth + 1,
        is_intermediate=True,
        operations=tuple(pre_ops),
    )
    # The metal salts are the buildable inputs consumed to form the intermediate.
    for precursor in recipe.precursors:
        in_stock = is_in_stock(precursor)
        intermediate.children.append(
            DAGNode(
                precursor.formula,
                node.modality,
                node.depth + 2,
                in_stock=in_stock,
                dangling=not in_stock,
            )
        )
    # This node now consumes the single intermediate via calcination.
    node.children = [intermediate]
    node.operations = tuple(calcine_ops)
    return node


def insert_solution_intermediates(dag: SynthesisDAG, is_in_stock: StockTest) -> SynthesisDAG:
    """Apply :func:`insert_solution_intermediate` to every recipe node in the DAG."""
    if dag.modality not in {"hydrothermal", "precipitation"}:
        return dag

    def _walk(node: DAGNode) -> None:
        # Only split nodes that still hold their own salt-precursor recipe (i.e.
        # have not already been restructured).
        if node.recipe is not None and not (len(node.children) == 1 and node.children[0].is_intermediate):
            insert_solution_intermediate(node, is_in_stock)
        for child in node.children:
            if not child.is_intermediate:
                _walk(child)

    _walk(dag.root)
    return dag
