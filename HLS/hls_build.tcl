# This script sets up and compiles an HLS project.
#
# vitis_hls hls_build.tcl
# vitis_hls -p hls_proj/hls.app

open_project -reset fsk_cnn_demod
set_top fsk_cnn_demod
add_files fsk_cnn_demod.hpp
add_files fsk_cnn_demod.cpp
add_files -tb fsk_cnn_demod_tb.cpp
open_solution -reset "solution1"
set_part {xc7a100t-csg324-1}
create_clock -period 10 -name default
config_export -format ip_catalog -rtl verilog
csynth_design
csim_design -clean
cosim_design
export_design -rtl verilog -format ip_catalog -description "fsk_cnn_demod." -vendor "caccolillo" -display_name "fsk_cnn_demod" -output "../fsk_cnn_demod.zip"
exit
