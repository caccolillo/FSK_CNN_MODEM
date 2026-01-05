#include "fsk_cnn_demod.hpp"


// Top-level function for HLS synthesis
void fsk_cnn_demod(
    fixed_t input_I[SAMPLES_PER_SYMBOL],
    fixed_t input_Q[SAMPLES_PER_SYMBOL],
    fixed_t output[FC_OUT],
    int *predicted_bit
) {
#pragma HLS INTERFACE mode=s_axilite port=return
#pragma HLS INTERFACE mode=s_axilite port=predicted_bit
#pragma HLS INTERFACE mode=m_axi depth=8 port=input_I offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi depth=8 port=input_Q offset=slave bundle=gmem1
#pragma HLS INTERFACE mode=m_axi depth=2 port=output offset=slave bundle=gmem2

    // Intermediate arrays
    fixed_t conv_out[SAMPLES_PER_SYMBOL];
    fixed_t pooled[POOLED_SIZE];
    
#pragma HLS ARRAY_PARTITION variable=conv_out complete dim=1
#pragma HLS ARRAY_PARTITION variable=pooled complete dim=1
    
    // Layer 1: Convolution (1x1 kernel, pointwise operation)
    CONV_LOOP: for (int i = 0; i < SAMPLES_PER_SYMBOL; i++) {
#pragma HLS PIPELINE II=1
        fixed_t sum = CONV_BIAS[0];
        sum += input_I[i] * CONV_WEIGHT[0][0];
        sum += input_Q[i] * CONV_WEIGHT[0][1];
        conv_out[i] = sum;
    }
    
    // Layer 2: MaxPooling (2x2 window, stride 2)
    POOL_LOOP: for (int i = 0; i < POOLED_SIZE; i++) {
#pragma HLS PIPELINE II=1
        int idx1 = i * 2;
        int idx2 = i * 2 + 1;
        pooled[i] = (conv_out[idx1] > conv_out[idx2]) ? conv_out[idx1] : conv_out[idx2];
    }
    
    // Layer 3: Fully Connected Layer
    FC_LOOP: for (int out = 0; out < FC_OUT; out++) {
#pragma HLS PIPELINE II=1
        fixed_t sum = FC_BIAS[out];
        for (int in = 0; in < FC_IN; in++) {
            sum += pooled[in] * FC_WEIGHT[out][in];
        }
        output[out] = sum;
    }
    
    // Determine predicted bit (argmax)
    *predicted_bit = (output[1] > output[0]) ? 1 : 0;
}
