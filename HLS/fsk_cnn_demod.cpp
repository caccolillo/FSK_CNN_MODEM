#include "fsk_cnn_demod.hpp"
#include <hls_math.h>

// Optimized helper functions for dataflow

static void read_iq_parallel(
    hls::stream<axis_data> &input_I,
    hls::stream<axis_data> &input_Q,
    fixed_t I_buffer[SAMPLES_PER_SYMBOL],
    fixed_t Q_buffer[SAMPLES_PER_SYMBOL]
) {
#pragma HLS INLINE off
    // Read I and Q samples in parallel with II=1
    READ_IQ: for (int i = 0; i < SAMPLES_PER_SYMBOL; i++) {
#pragma HLS PIPELINE II=1
        axis_data i_data = input_I.read();
        axis_data q_data = input_Q.read();
        I_buffer[i] = i_data.data;
        Q_buffer[i] = q_data.data;
    }
}

static void compute_convolution(
    fixed_t I_buffer[SAMPLES_PER_SYMBOL],
    fixed_t Q_buffer[SAMPLES_PER_SYMBOL],
    acc_t conv_out[SAMPLES_PER_SYMBOL]
) {
#pragma HLS INLINE off
    
    // Precompute scaled weights (constant folding optimization)
    const acc_t w_i = (acc_t)CONV_WEIGHT_QUANT[0][0] * CONV_WEIGHT_SCALE;
    const acc_t w_q = (acc_t)CONV_WEIGHT_QUANT[0][1] * CONV_WEIGHT_SCALE;
    const acc_t bias = (acc_t)CONV_BIAS_QUANT[0] * CONV_BIAS_SCALE;
    
    CONV: for (int t = 0; t < SAMPLES_PER_SYMBOL; t++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=2
        acc_t sum = bias;
        sum += (acc_t)I_buffer[t] * w_i;
        sum += (acc_t)Q_buffer[t] * w_q;
        conv_out[t] = sum;
    }
}

static void compute_pooling(
    acc_t conv_out[SAMPLES_PER_SYMBOL],
    fixed_t pooled[POOLED_SIZE]
) {
#pragma HLS INLINE off
    
    POOL: for (int i = 0; i < POOLED_SIZE; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=2
        int idx = i << 1;  // i * 2
        pooled[i] = (conv_out[idx] > conv_out[idx + 1]) ?
                     (fixed_t)conv_out[idx] :
                     (fixed_t)conv_out[idx + 1];
    }
}

static void compute_fc(
    fixed_t pooled[POOLED_SIZE],
    acc_t fc_out[FC_OUT]
) {
#pragma HLS INLINE off
    
    // Precompute scaled biases
    const acc_t bias_0 = (acc_t)FC_BIAS_QUANT[0] * FC_BIAS_SCALE;
    const acc_t bias_1 = (acc_t)FC_BIAS_QUANT[1] * FC_BIAS_SCALE;
    
    // Fully unrolled FC computation (only 2 outputs, 4 inputs)
    acc_t acc_0 = bias_0;
    acc_t acc_1 = bias_1;
    
    FC_COMPUTE: for (int i = 0; i < FC_IN; i++) {
#pragma HLS UNROLL
        acc_t weight_0 = (acc_t)FC_WEIGHT_QUANT[0][i] * FC_WEIGHT_SCALE;
        acc_t weight_1 = (acc_t)FC_WEIGHT_QUANT[1][i] * FC_WEIGHT_SCALE;
        
        acc_0 += (acc_t)pooled[i] * weight_0;
        acc_1 += (acc_t)pooled[i] * weight_1;
    }
    
    fc_out[0] = acc_0;
    fc_out[1] = acc_1;
}

static void output_decision(
    acc_t fc_out[FC_OUT],
    hls::stream<axis_output> &predicted_bit
) {
#pragma HLS INLINE off
    
    axis_output out;
    out.data = (ap_uint<1>)((fc_out[1] > fc_out[0]) ? 1 : 0);
    out.last = 1;
    predicted_bit.write(out);
}

void fsk_cnn_demod(
    hls::stream<axis_data> &input_I,
    hls::stream<axis_data> &input_Q,
    hls::stream<axis_output> &predicted_bit
) {
#pragma HLS INTERFACE axis port=input_I
#pragma HLS INTERFACE axis port=input_Q
#pragma HLS INTERFACE axis port=predicted_bit
#pragma HLS INTERFACE ap_ctrl_none port=return

// Enable task-level pipelining - allows new input while processing previous
#pragma HLS DATAFLOW

    // Intermediate buffers
    fixed_t I_buffer[SAMPLES_PER_SYMBOL];
    fixed_t Q_buffer[SAMPLES_PER_SYMBOL];
    acc_t conv_out[SAMPLES_PER_SYMBOL];
    fixed_t pooled[POOLED_SIZE];
    acc_t fc_out[FC_OUT];

#pragma HLS ARRAY_PARTITION variable=I_buffer complete
#pragma HLS ARRAY_PARTITION variable=Q_buffer complete
#pragma HLS ARRAY_PARTITION variable=conv_out complete
#pragma HLS ARRAY_PARTITION variable=pooled complete
#pragma HLS ARRAY_PARTITION variable=fc_out complete

    // Dataflow pipeline stages
    read_iq_parallel(input_I, input_Q, I_buffer, Q_buffer);
    compute_convolution(I_buffer, Q_buffer, conv_out);
    compute_pooling(conv_out, pooled);
    compute_fc(pooled, fc_out);
    output_decision(fc_out, predicted_bit);
}