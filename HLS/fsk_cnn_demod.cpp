#include "fsk_cnn_demod.hpp"
#include <hls_math.h>

void fsk_cnn_demod(
    hls::stream<axis_data> &input_I,
    hls::stream<axis_data> &input_Q,
    hls::stream<axis_output> &predicted_bit
) {
#pragma HLS INTERFACE axis port=input_I
#pragma HLS INTERFACE axis port=input_Q
#pragma HLS INTERFACE axis port=predicted_bit
#pragma HLS INTERFACE s_axilite port=return

    // Intermediate buffers
    acc_t conv_out[SAMPLES_PER_SYMBOL];
    fixed_t pooled[POOLED_SIZE];
    acc_t fc_out[FC_OUT];

    fixed_t I_buffer[SAMPLES_PER_SYMBOL];
    fixed_t Q_buffer[SAMPLES_PER_SYMBOL];

#pragma HLS ARRAY_PARTITION variable=conv_out complete
#pragma HLS ARRAY_PARTITION variable=pooled complete
#pragma HLS ARRAY_PARTITION variable=fc_out complete
#pragma HLS ARRAY_PARTITION variable=I_buffer complete
#pragma HLS ARRAY_PARTITION variable=Q_buffer complete

    // ===== Read I samples =====
    READ_I: for (int i = 0; i < SAMPLES_PER_SYMBOL; i++) {
#pragma HLS PIPELINE
        axis_data t = input_I.read();
        I_buffer[i] = t.data;
        if (t.last) break;
    }

    // ===== Read Q samples =====
    READ_Q: for (int i = 0; i < SAMPLES_PER_SYMBOL; i++) {
#pragma HLS PIPELINE
        axis_data t = input_Q.read();
        Q_buffer[i] = t.data;
        if (t.last) break;
    }

    // ===== CONVOLUTION (Quantized) =====
    // Matches Python: conv_sum += symbol[i, t] * conv_w[0, i, 0] * scale_w
    CONV: for (int t = 0; t < SAMPLES_PER_SYMBOL; t++) {
#pragma HLS PIPELINE II=1
        acc_t sum = 0;

        // Apply quantized weights with scaling
        // I channel (index 0): weight = -15, scale = 0.083095
        acc_t w_i = (acc_t)CONV_WEIGHT_QUANT[0][0] * CONV_WEIGHT_SCALE;
        sum += (acc_t)I_buffer[t] * w_i;

        // Q channel (index 1): weight = 15, scale = 0.083095
        acc_t w_q = (acc_t)CONV_WEIGHT_QUANT[0][1] * CONV_WEIGHT_SCALE;
        sum += (acc_t)Q_buffer[t] * w_q;

        // Add bias (effectively zero: 0 * 0.0 = 0)
        acc_t b = (acc_t)CONV_BIAS_QUANT[0] * CONV_BIAS_SCALE;
        sum += b;

        conv_out[t] = sum;
    }

    // ===== MAX POOLING (kernel=2, stride=2) =====
    POOL: for (int i = 0; i < POOLED_SIZE; i++) {
#pragma HLS PIPELINE II=1
        int idx = i << 1;  // i * 2
        pooled[i] = (conv_out[idx] > conv_out[idx + 1]) ?
                     (fixed_t)conv_out[idx] :
                     (fixed_t)conv_out[idx + 1];
    }

    // ===== FULLY CONNECTED (Quantized) =====
    // Matches Python: s += pool_out[i] * fc_w[j, i] * scale_w_fc
    FC: for (int j = 0; j < FC_OUT; j++) {
#pragma HLS PIPELINE II=1
        acc_t acc = 0;

        // Add bias first
        acc_t bias = (acc_t)FC_BIAS_QUANT[j] * FC_BIAS_SCALE;
        acc += bias;

        // Accumulate weighted inputs
        for (int i = 0; i < FC_IN; i++) {
            acc_t weight = (acc_t)FC_WEIGHT_QUANT[j][i] * FC_WEIGHT_SCALE;
            acc += (acc_t)pooled[i] * weight;
        }

        fc_out[j] = acc;
    }

    // ===== OUTPUT DECISION (Argmax - no softmax needed for inference) =====
    // Python: pred_bit = np.argmax(probs), which is equivalent to fc_out[1] > fc_out[0]
    axis_output out;
    out.data = (ap_uint<1>)((fc_out[1] > fc_out[0]) ? 1 : 0);
    out.last = 1;
    predicted_bit.write(out);
}
