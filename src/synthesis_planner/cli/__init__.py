"""
Command-line interface for the synthesis planner.

Re-exports the argparse app so ``synthesis_planner.cli`` continues to expose
``build_parser``, ``load_config``, and ``main`` (the ``mcts-plan`` entry point
and the behavioral contract in ``tests/test_cli.py``).

© 2026. Triad National Security, LLC. All rights reserved.
"""

from .main import build_parser, load_config, main

__all__ = ["build_parser", "load_config", "main"]
