#!/usr/bin/gnuplot

# Top 10 Compounds by Composite Score
# Composite score shown as stacked horizontal bars colored by component
# Components: ehull_reward (orange), 2.5*r_DOS (green)
# Stars indicate target compounds (V6Sn6U, Nb6Sn6U, Cr6Ge6U, Co6Ge6U)
# Weights: beta=1.0, gamma=2.5

reset

# Output settings - 3.25in x 3.25in (single panel)
set terminal pdfcairo enhanced font "Helvetica,8" size 3.25in,3.25in
set output 'top10_by_composite.pdf'

# Color definitions (consistent with other figures)
ehull_color = "#ff7f0e"   # Orange for ehull_reward
dos_color = "#2ca02c"     # Green for r_DOS

set xlabel "Composite Score" font ",8" offset 0,0.5
unset ylabel

set yrange [0.5:10.5]
set xrange [0:*]
set ytics font ",8"
set xtics font ",8"

# Function to format compound names with subscripts
format_formula(f) = system(sprintf("python format_formula_labels.py '%s'", f))

# Read labels for y-axis
set ytics ('' 1)
do for [i=1:10] {
    formula = system(sprintf("awk 'NR==%d {print $2}' gnuplot_data/top10_labels.dat", i))
    formatted = format_formula(formula)
    set ytics add (formatted i)
}

# No grid
unset grid

# Key/legend for components (positioned at top right)
set key at graph 0.95, graph 0.90 right top font ",8" spacing 1

# Plot stacked horizontal bars with stars for target compounds
# ehull_reward: 0 to beta*ehull_reward
# gamma*r_DOS: beta*ehull_reward to composite_score

set style fill solid 0.7 noborder
bar_height = 0.4

# Star marker settings
star_offset = 0.15

beta = 1.0
gamma = 2.5

plot 'gnuplot_data/top10_components.dat' \
    using ($2*beta/2):1:(0):($2*beta):($1-bar_height/2):($1+bar_height/2) \
    with boxxyerrorbars lc rgb ehull_color fs solid 0.7 notitle, \
    '' using (($2*beta+$3*gamma/2)):1:($2*beta):(($2*beta+$3*gamma)):($1-bar_height/2):($1+bar_height/2) \
    with boxxyerrorbars lc rgb dos_color fs solid 0.7 notitle, \
    '' using ($2*beta+$3*gamma+star_offset):1:($4 == 1 ? "*" : "") \
    with labels left font ",16" tc rgb "black" notitle, \
    NaN with boxxyerrorbars lc rgb ehull_color fs solid 0.7 title "{/:Italic E}_{Hull}", \
    NaN with boxxyerrorbars lc rgb dos_color fs solid 0.7 title "{/:Italic r}_{DOS}"

print "Generated: top10_by_composite.pdf"
