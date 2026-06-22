#!/usr/bin/gnuplot

# Best composite reward as a function of iteration
# Composite = 1.0*ehull_reward + 2.5*r_DOS

set terminal pdfcairo enhanced color font "Arial,10" size 3.25in,3.25in

set output "composite_convergence.pdf"

unset title
set xlabel "Iteration" font "Arial,10"
set ylabel "Best Composite Score" font "Arial,10"

# No grid
unset grid

set lmargin at screen 0.15
set rmargin at screen 0.95
set tmargin at screen 0.92
set bmargin at screen 0.12

set autoscale

# Dashed line at y=0
set arrow from graph 0,first 0 to graph 1,first 0 nohead lc rgb "#888888" lw 1.0 dt (5,5)

# Plot convergence
plot 'composite_convergence.dat' using 1:2 \
    with lines lw 2 lc rgb "#1f77b4" notitle

set output
