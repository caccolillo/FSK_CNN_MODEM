#include "fsk_cnn_demod.hpp"


// Top-level function for HLS synthesis
void fsk_cnn_demod(
    hls::stream<fixed_t> &input_I,
    hls::stream<fixed_t> &input_Q,
    hls::stream<int> &predicted_bit
) {
#pragma HLS INTERFACE mode=axis port=input_I
#pragma HLS INTERFACE mode=axis port=input_Q
#pragma HLS INTERFACE mode=axis port=predicted_bit
#pragma HLS INTERFACE mode=s_axilite port=return

    // Intermediate arrays
    fixed_t conv_out[SAMPLES_PER_SYMBOL];
    fixed_t pooled[POOLED_SIZE];
    fixed_t output[POOLED_SIZE];

    // Buffer to store input samples
    fixed_t I_buffer[SAMPLES_PER_SYMBOL];
    fixed_t Q_buffer[SAMPLES_PER_SYMBOL];

#pragma HLS ARRAY_PARTITION variable=conv_out complete dim=1
#pragma HLS ARRAY_PARTITION variable=pooled complete dim=1
#pragma HLS ARRAY_PARTITION variable=I_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=Q_buffer complete dim=1

    // Read input streams into buffers
    READ_I_LOOP: for (int i = 0; i < SAMPLES_PER_SYMBOL; i++) {
#pragma HLS PIPELINE II=1
        I_buffer[i] = input_I.read();
    }

    READ_Q_LOOP: for (int i = 0; i < SAMPLES_PER_SYMBOL; i++) {
#pragma HLS PIPELINE II=1
        Q_buffer[i] = input_Q.read();
    }
    
    // Layer 1: Convolution (1x1 kernel, pointwise operation)
    CONV_LOOP: for (int i = 0; i < SAMPLES_PER_SYMBOL; i++) {
#pragma HLS PIPELINE II=1
        fixed_t sum = CONV_BIAS[0];
        sum += I_buffer[i] * CONV_WEIGHT[0][0];
        sum += Q_buffer[i] * CONV_WEIGHT[0][1];
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
    
    // Determine predicted bit (argmax) and write to output stream
    int result = (output[1] > output[0]) ? 1 : 0;
    predicted_bit.write(result);
}
