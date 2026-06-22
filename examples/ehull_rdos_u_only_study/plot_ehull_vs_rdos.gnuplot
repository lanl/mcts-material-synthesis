#!/usr/bin/gnuplot

# Set terminal to PDF - 3.25in x 3.25in, 10pt font
set terminal pdfcairo enhanced color font "Arial,10" size 3.25in,3.25in

# Output file
set output "ehull_vs_rdos.pdf"

# Title and labels
unset title
set xlabel "{/:Italic r}_{DOS}" font "Arial,10"
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

# Set axis ranges
set autoscale

# Remove zero axes
unset xzeroaxis
unset yzeroaxis

# Shade the region where E_hull < 0 (thermodynamically stable) with light pastel green
# This is the bottom strip below y=0
set object rectangle from graph 0,graph 0 to graph 1,first 0 fillcolor rgb "#E8F5E8" fillstyle solid noborder behind

# Add dashed reference line at y=0 (E_hull = 0)
set arrow from graph 0,first 0 to graph 1,first 0 nohead lc rgb "#000000" lw 1.5 dt (5,5)

# Plot the data
# Column layout: e_hull  2.5*r_dos  name
# Note: x=column 2 (2.5*r_DOS), y=column 1 (e_hull)
plot 'gnuplot_data_ehull_rdos/all_u_compounds.dat' using 2:($1 <= 2 ? $1 : 1/0) \
    with points pt 7 ps 0.3 lc rgb "#D0D0D0" title "All Compounds", \
    'gnuplot_data_ehull_rdos/synthesized_compounds.dat' using 2:1 \
    with points pt 5 ps 0.7 lc rgb "#9467BD" title "Successful Synthesis", \
    'gnuplot_data_ehull_rdos/unsynthesized_compounds.dat' using 2:1 \
    with points pt 4 ps 0.7 lc rgb "#9467BD" title "Unsuccessful Synthesis", \
    'gnuplot_data_ehull_rdos/top10_mcts.dat' using 2:1 \
    with points pt 8 ps 0.9 lc rgb "#17BECF" title "Top 10 MCTS Predictions"

# Reset
set output
