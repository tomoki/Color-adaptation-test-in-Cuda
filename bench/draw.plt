set terminal pdf
set output "plot.pdf"
set title "color adaptation test"
set key left top
set ylabel "mean of 4 times run, discarding first run [us]"
set xlabel "size [/512*512px]"

plot   "k.txt" using 1:2 title "CUDA uchar4"       w lp, \
       "k.txt" using 1:3 title "CPU"               w lp, \
       "k.txt" using 1:4 title "CUDA uchar array"  w lp
