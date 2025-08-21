# Gnuplot script for constellation plotting
set terminal pdf enhanced color size 12,6
set output 'constellation_comparison.pdf'
set multiplot layout 1,2

# Input constellation
set title 'Input Constellation (Transmitted)'
set xlabel 'Real Part'
set ylabel 'Imaginary Part'
set grid
set size square
set xrange [-5:5]
set yrange [-5:5]
plot 'input_constellation.dat' using 1:2 with points pointtype 7 pointsize 1.2 linecolor rgb 'dark-blue' title 'Input Symbols'

# Output constellation
set title 'Output Constellation (Received & Equalized)'
set xlabel 'Real Part'
set ylabel 'Imaginary Part'
set grid
set size square
set xrange [-5:5]
set yrange [-5:5]
plot 'output_constellation.dat' using 1:2 with points pointtype 7 pointsize 1.2 linecolor rgb 'dark-blue' title 'Received Symbols'

unset multiplot
