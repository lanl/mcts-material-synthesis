#!/bin/bash
# Generate all figures for the ehull_rdos U-only study (beta=1.0, gamma=2.5)
# Run this from inside examples/ehull_rdos_u_only_study/ after run_study.sh
# has produced all_compounds.csv, convergence_history.csv, and mcts_object.pkl here.

set -e

echo "=================================================="
echo "Generating figures for the ehull_rdos U-only study"
echo "Weights: beta=1.0, gamma=2.5"
echo "=================================================="

echo ""
echo "Step 1: Computing composite scores from all_compounds.csv"
python generate_top10_report.py

echo ""
echo "Step 2: Preparing top 10 plot data"
python prepare_top10_plot_data.py

echo ""
echo "Step 3: Generating top10_by_composite.pdf"
gnuplot plot_top10_ehull_rdos_bars_stacked.gnuplot

echo ""
echo "Step 4: Preparing SG191 comparison data (requires shunshun_mace_predictions_with_elements.csv - see README)"
python prepare_sg191_composite_data.py

echo ""
echo "Step 5: Generating SG191 comparison scatter plot"
gnuplot plot_sg191_comparison_no_pareto.gnuplot

echo ""
echo "Step 6: Generating E_hull vs r_DOS scatter plot (ehull_vs_rdos.pdf)"
python prepare_ehull_vs_rdos_data.py
gnuplot plot_ehull_vs_rdos.gnuplot

echo ""
echo "Step 7: Generating top10_ehull_rdos_bars.pdf"
python prepare_top10_bar_data.py
gnuplot plot_top10_ehull_rdos_bars.gnuplot

echo ""
echo "Step 8: Generating composite_convergence.pdf"
python prepare_composite_convergence.py
gnuplot plot_composite_convergence.gnuplot

echo ""
echo "Step 9: Generating radial_tree_composite.png"
python create_composite_radial_tree.py

echo ""
echo "=================================================="
echo "ALL FIGURES GENERATED"
echo "=================================================="
