#!/usr/bin/gnuplot

# Top 10 compounds: E_hull and 2.5*r_DOS as grouped vertical bars
# Sorted by composite score (same order as top10_by_modified_composite.pdf)
# Stars indicate target compounds (V6Sn6U, Nb6Sn6U, Cr6Ge6U, Co6Ge6U)

reset

# Output settings - 3.25in x 3.25in
set terminal pdfcairo enhanced font "Helvetica,8" size 3.25in,3.25in
set output 'top10_ehull_rdos_bars.pdf'

# Color definitions (consistent with other figures)
ehull_color = "#ff7f0e"   # Orange for E_hull
dos_color = "#2ca02c"     # Green for 2.5*r_DOS

# Axis labels
unset xlabel
set ylabel "{/:Italic E}_{hull} (eV/atom) / {/:Italic r}_{DOS}" font ",8" offset 1.5,0

# Y range: put y=0 in lower portion
set yrange [-0.1:0.85]
set ytics font ",8"

# X range
set xrange [0.25:10.75]

# Function to format compound names with subscripts
format_formula(f) = system(sprintf("python format_formula_labels.py '%s'", f))

# Raise bottom margin so rotated labels don't overlap the figure
set bmargin at screen 0.18

# Read labels for x-axis (vertical, centered on tick marks)
set xtics rotate by -90 center font ",10" offset 0,-3
set xtics ('' 1)
do for [i=1:10] {
    formula = system(sprintf("awk 'NR==%d {print $2}' gnuplot_data_bars/top10_bar_labels.dat", i))
    formatted = format_formula(formula)
    set xtics add (formatted i)
}

# No grid
unset grid

# Dashed line at y=0
set arrow from graph 0,first 0 to graph 1,first 0 nohead lc rgb "#000000" lw 1.0 dt (5,5)

# Key/legend
set key at graph 0.95, graph 0.95 right top font ",8" spacing 1.2

# Bar width and offset for grouped bars (touching, no gap)
bar_w = 0.3
offset = bar_w / 2.0  # bars touch: each shifted by half its width

set boxwidth bar_w

# Data columns: rank e_hull weighted_rdos is_target name
# E_hull bars: solid fill
# r_DOS bars: solid fill + striped pattern overlay
plot 'gnuplot_data_bars/top10_ehull_rdos_bars.dat' \
    using ($1 - offset):2:(bar_w) with boxes lc rgb ehull_color fs solid 0.7 noborder title "{/:Italic E}_{hull}", \
    '' using ($1 + offset):3:(bar_w) with boxes lc rgb dos_color fs solid 0.7 noborder title "{/:Italic r}_{DOS}", \
    '' using 1:(($2 < $3 ? $3 : $2) + 0.03):($4 == 1 ? "*" : "") \
    with labels center font ",14" tc rgb "black" notitle

print "Generated: top10_ehull_rdos_bars.pdf"
