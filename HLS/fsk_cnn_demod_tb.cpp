#ifndef __SYNTHESIS__
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>

#include "fsk_cnn_demod.hpp"

void generate_fsk_symbol(int bit, float input_I[8], float input_Q[8]) {
    // Match Python FSK generation exactly:
    // phase_shift = 2 * np.pi * (1 if bit == 1 else -1) / SAMPLES_PER_SYMBOL
    // I = np.cos(phase_shift * np.arange(SAMPLES_PER_SYMBOL))
    // Q = np.sin(phase_shift * np.arange(SAMPLES_PER_SYMBOL))

    float phase_shift;
    if (bit == 1) {
        phase_shift = 2.0 * M_PI * 1.0 / 8.0;  // 2*pi/8 for bit 1
    } else {
        phase_shift = 2.0 * M_PI * (-1.0) / 8.0;  // -2*pi/8 for bit 0
    }

    for (int i = 0; i < 8; i++) {
        input_I[i] = cos(phase_shift * i);
        input_Q[i] = sin(phase_shift * i);
    }
}

int main() {
    // Use fixed seed for reproducibility during debug
    srand(42);

    std::cout << "=================================================\n";
    std::cout << "FSK CNN DEMODULATOR - DEBUG TESTBENCH\n";
    std::cout << "Testing with exact Python FSK generation\n";
    std::cout << "=================================================\n\n";

    int correct_count = 0;
    int total_bits = 10;

    // Test with 10 random bits
    for (int test_num = 0; test_num < total_bits; test_num++) {
        // Generate random bit (0 or 1)
        int test_bit = rand() % 2;

        std::cout << "Test #" << (test_num + 1) << " - Bit = " << test_bit << "\n";
        std::cout << "-------------------------------------------------\n";

        // Generate FSK symbol
        float I_float[8], Q_float[8];
        generate_fsk_symbol(test_bit, I_float, Q_float);

        // Print generated symbols for debugging
        std::cout << "Generated I/Q samples:\n";
        std::cout << std::fixed << std::setprecision(6);
        for (int i = 0; i < 8; i++) {
            std::cout << "  t=" << i << ": I=" << I_float[i]
                      << ", Q=" << Q_float[i] << "\n";
        }

        // Create AXI streams
        hls::stream<axis_data> input_I_stream;
        hls::stream<axis_data> input_Q_stream;
        hls::stream<axis_output> predicted_bit_stream;

        // Convert to fixed-point and write to streams
        for (int i = 0; i < 8; i++) {
            axis_data I_val, Q_val;
            I_val.data = fixed_t(I_float[i]);
            I_val.last = (i == 7) ? 1 : 0;
            Q_val.data = fixed_t(Q_float[i]);
            Q_val.last = (i == 7) ? 1 : 0;
            input_I_stream.write(I_val);
            input_Q_stream.write(Q_val);
        }

        // Run inference
        fsk_cnn_demod(input_I_stream, input_Q_stream, predicted_bit_stream);

        // Read predicted bit from output stream
        axis_output predicted_output = predicted_bit_stream.read();
        int predicted_bit = predicted_output.data;

        // Check if prediction is correct
        bool is_correct = (predicted_bit == test_bit);
        if (is_correct) {
            correct_count++;
        }

        // Print results
        std::cout << "Actual bit:    " << test_bit << "\n";
        std::cout << "Predicted bit: " << predicted_bit
                  << " (tlast=" << predicted_output.last << ")\n";
        std::cout << "Result: " << (is_correct ? "✓ PASS" : "✗ FAIL") << "\n\n";
    }

    // Print summary statistics
    std::cout << "=================================================\n";
    std::cout << "TEST SUMMARY\n";
    std::cout << "=================================================\n";
    std::cout << "Total bits tested: " << total_bits << "\n";
    std::cout << "Correct predictions: " << correct_count << "\n";
    std::cout << "Incorrect predictions: " << (total_bits - correct_count) << "\n";
    std::cout << "Accuracy: " << (100.0 * correct_count / total_bits) << "%\n";
    std::cout << "=================================================\n";

    if (correct_count == total_bits) {
        std::cout << "✓ ALL TESTS PASSED!\n";
    } else {
        std::cout << "✗ SOME TESTS FAILED\n";
    }
    std::cout << "=================================================\n";

    return (correct_count == total_bits) ? 0 : 1;
}
#endif
