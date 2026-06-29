#!/bin/bash
# Generate all figures for the gamma-normalized ehull_rdos U-only study
# (beta=1.0, gamma=1/2516.1664410449775 - fixed in generate_figures.py's
# NORMALIZED_GAMMA, NOT read from config.json; see run_study.sh)
# Run this from inside this directory after run_study.sh has produced
# all_compounds.csv, convergence_history.csv, and mcts_object.pkl here.

set -e

echo "=================================================="
echo "Generating figures for the gamma-normalized ehull_rdos U-only study"
echo "Weights: beta=1.0, gamma=0.00039742998860786596 (fixed, not from config.json)"
echo "=================================================="

echo ""

echo "Step: Generating all figures (PNG) using Python plotting pipeline"
python generate_figures.py

echo ""
echo "=================================================="
echo "ALL FIGURES GENERATED"
echo "=================================================="
