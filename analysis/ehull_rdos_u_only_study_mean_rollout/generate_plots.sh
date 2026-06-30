#!/bin/bash
# Generate all figures for the mean-rollout-aggregation variant of the
# ehull_rdos U-only study (beta/gamma from config.json, default 1.0/0.0001 -
# see config.example.json; gamma is unchanged from the calibrated study, only
# --rollout-aggregation differs - see run_study.sh)
# Run this from inside this directory after run_study.sh has produced
# all_compounds.csv, convergence_history.csv, and mcts_object.pkl here.

set -e

echo "=================================================="
echo "Generating figures for the mean-rollout-aggregation ehull_rdos U-only study"
echo "Weights: beta/gamma from config.json (default beta=1.0, gamma=0.0001)"
echo "=================================================="

echo ""

echo "Step: Generating all figures (PNG) using Python plotting pipeline"
python generate_figures.py

echo ""
echo "=================================================="
echo "ALL FIGURES GENERATED"
echo "=================================================="
