#include <ap_fixed.h>
#include <hls_stream.h>
#include <ap_int.h>

// Fixed-point types
typedef ap_fixed<16, 8> fixed_t;     // For input/intermediate values
typedef ap_fixed<48, 24> acc_t;      // Wider accumulator for precision
typedef ap_int<8> quant_t;           // For quantized weights

// AXI Stream structure with tlast
struct axis_data {
    fixed_t data;
    ap_uint<1> last;
};

struct axis_output {
    ap_uint<1> data;
    ap_uint<1> last;
};

// Model parameters
const int IN_CHANNELS = 2;
const int OUT_CHANNELS = 1;
const int SAMPLES_PER_SYMBOL = 8;
const int POOLED_SIZE = 4;
const int FC_IN = 4;
const int FC_OUT = 2;

// ===== QUANTIZED PARAMETERS (from quantized_params.txt) =====

// Conv layer: quantized weights (int8) and scales
const quant_t CONV_WEIGHT_QUANT[OUT_CHANNELS][IN_CHANNELS] = {
    {-15, 15}
};
const fixed_t CONV_WEIGHT_SCALE = 0.083095;

const quant_t CONV_BIAS_QUANT[OUT_CHANNELS] = {0};
const fixed_t CONV_BIAS_SCALE = 0.0;  // Bias is effectively zero

// FC layer: quantized weights (int8) and scales
const quant_t FC_WEIGHT_QUANT[FC_OUT][FC_IN] = {
    {-2, -14, 3, 15},
    {8, 15, -8, -15}
};
const fixed_t FC_WEIGHT_SCALE = 0.093284;

const quant_t FC_BIAS_QUANT[FC_OUT] = {-15, 15};
const fixed_t FC_BIAS_SCALE = 0.000619;

// Function declaration
void fsk_cnn_demod(
    hls::stream<axis_data> &input_I,
    hls::stream<axis_data> &input_Q,
    hls::stream<axis_output> &predicted_bit
);
