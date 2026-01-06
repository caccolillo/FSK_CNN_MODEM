#ifndef __SYNTHESIS__
#include <iostream>
#include <cmath>

#include "fsk_cnn_demod.hpp"

void generate_fsk_symbol(float bit, float input_I[8], float input_Q[8]) {
    const float F_DEVIATION = 1e6;
    const float F_SAMPLING = 8e6;

    float phase_shift;
    if (bit == 0) {
        phase_shift = 2 * M_PI * (-F_DEVIATION) / F_SAMPLING;
    } else {
        phase_shift = 2 * M_PI * F_DEVIATION / F_SAMPLING;
    }

    for (int i = 0; i < 8; i++) {
        input_I[i] = cos(phase_shift * i);
        input_Q[i] = sin(phase_shift * i);
    }
}

int main() {
    std::cout << "=================================================\n";
    std::cout << "FSK CNN DEMODULATOR - VITIS HLS TESTBENCH\n";
    std::cout << "=================================================\n\n";

    // Test with both bit values
    for (int test_bit = 0; test_bit <= 1; test_bit++) {
        std::cout << "Testing with bit = " << test_bit << "\n";
        std::cout << "-------------------------------------------------\n";

        // Generate FSK symbol
        float I_float[8], Q_float[8];
        generate_fsk_symbol(test_bit, I_float, Q_float);

        // Create AXI streams
        hls::stream<fixed_t> input_I_stream;
        hls::stream<fixed_t> input_Q_stream;
        hls::stream<int> predicted_bit_stream;

        // Convert to fixed-point and write to streams
        std::cout << "Input I/Q samples:\n";
        for (int i = 0; i < 8; i++) {
            fixed_t I_val = fixed_t(I_float[i]);
            fixed_t Q_val = fixed_t(Q_float[i]);
            input_I_stream.write(I_val);
            input_Q_stream.write(Q_val);
            std::cout << "  [" << i << "] I=" << I_val.to_float()
                      << ", Q=" << Q_val.to_float() << "\n";
        }

        // Run inference
        fsk_cnn_demod(input_I_stream, input_Q_stream, predicted_bit_stream);

        // Read predicted bit from output stream
        int predicted_bit = predicted_bit_stream.read();

        // Print results
        std::cout << "\nPredicted bit: " << predicted_bit << "\n";
        std::cout << "Actual bit:    " << test_bit << "\n";
        std::cout << "Match: " << (predicted_bit == test_bit ? "✓" : "✗") << "\n\n";
    }

    std::cout << "=================================================\n";
    std::cout << "Testbench complete!\n";
    std::cout << "=================================================\n";

    return 0;
}
#endif
