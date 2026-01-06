#include <ap_fixed.h>
#include <hls_stream.h>
#include <ap_int.h>

// Fixed-point type: 16-bit total, 8 fractional bits
typedef ap_fixed<16, 8> fixed_t;

// AXI Stream structure with tlast
struct axis_data {
    fixed_t data;
    ap_uint<1> last;
};

// Model parameters (from float_params.txt)
const int IN_CHANNELS = 2;
const int OUT_CHANNELS = 1;
const int SAMPLES_PER_SYMBOL = 8;
const int POOLED_SIZE = 4;
const int FC_IN = 4;
const int FC_OUT = 2;

// Conv layer parameters (1x1 kernel)
const fixed_t CONV_WEIGHT[OUT_CHANNELS][IN_CHANNELS] = {
    {fixed_t(0.00638873), fixed_t(2.4992344)}
};
const fixed_t CONV_BIAS[OUT_CHANNELS] = {fixed_t(-0.62816364)};

// FC layer parameters
const fixed_t FC_WEIGHT[FC_OUT][FC_IN] = {
    {fixed_t(-0.10768079), fixed_t(-1.1741734), fixed_t(0.3632769), fixed_t(1.4843116)},
    {fixed_t(0.86062443), fixed_t(1.5053908), fixed_t(-0.62919515), fixed_t(-1.29314)}
};
const fixed_t FC_BIAS[FC_OUT] = {fixed_t(0.22219384), fixed_t(0.24075383)};


// AXI Stream structure for output with tlast
struct axis_output {
    int data;
    ap_uint<1> last;
};


void fsk_cnn_demod(
    hls::stream<axis_data> &input_I,
    hls::stream<axis_data> &input_Q,
    hls::stream<axis_output> &predicted_bit
);
