#!/usr/bin/gnuplot

# Set terminal to PDF - 3.25in x 3.25in, 10pt font
set terminal pdfcairo enhanced color font "Arial,10" size 3.25in,3.25in

# Output file
set output "shunshun_compounds_comparison_sg191_no_pareto.pdf"

# Title and labels
unset title
set xlabel "{/:Italic E}_{form} (eV/atom)" font "Arial,10"
set ylabel "{/:Italic E}_{hull} (eV/atom)" font "Arial,10"

# No grid
unset grid

# Set key (legend) position - top left with vertical spacing, symbols on left
set key top left font "Arial,10" spacing 1.5 reverse Left

# Set margins for better layout
set lmargin at screen 0.12
set rmargin at screen 0.95
set tmargin at screen 0.92
set bmargin at screen 0.12

# Set axis ranges (will auto-scale, but can adjust if needed)
set autoscale

# Remove zero axes
unset xzeroaxis
unset yzeroaxis

# Shade the thermodynamically stable region (x<0, y<0) with light pastel green (no border)
set object rectangle from graph 0,graph 0 to first 0,first 0 fillcolor rgb "#E8F5E8" fillstyle solid noborder behind

# Add dashed reference lines at x=0 and y=0 (only to origin)
set arrow from first 0,graph 0 to first 0,first 0 nohead lc rgb "#000000" lw 1.5 dt (5,5)
set arrow from graph 0,first 0 to first 0,first 0 nohead lc rgb "#000000" lw 1.5 dt (5,5)

# Plot the data (without Pareto front)
plot 'gnuplot_data_sg191/all_compounds_sg191.dat' using 1:($2 <= 2 ? $2 : 1/0) \
    with points pt 7 ps 0.3 lc rgb "#D0D0D0" title "All Compounds", \
    '../shunshun_calculations/edit_synthesized_compounds.dat' using 1:2 \
    with points pt 5 ps 0.7 lc rgb "#9467BD" title "Successful Synthesis", \
    '../shunshun_calculations/edit_unsynthesized_compounds.dat' using 1:2 \
    with points pt 4 ps 0.7 lc rgb "#9467BD" title "Unsuccessful Synthesis", \
    'gnuplot_data_sg191/top10_mcts_composite.dat' using 1:($2 <= 2 ? $2 : 1/0) \
    with points pt 8 ps 0.9 lc rgb "#17BECF" title "Top 10 MCTS Predictions"

# Reset
set output
