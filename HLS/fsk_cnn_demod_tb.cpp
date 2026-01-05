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
        
        // Convert to fixed-point
        fixed_t input_I[8], input_Q[8];
        std::cout << "Input I/Q samples:\n";
        for (int i = 0; i < 8; i++) {
            input_I[i] = fixed_t(I_float[i]);
            input_Q[i] = fixed_t(Q_float[i]);
            std::cout << "  [" << i << "] I=" << input_I[i].to_float() 
                      << ", Q=" << input_Q[i].to_float() << "\n";
        }
        
        // Run inference
        fixed_t output[2];
        int predicted_bit;
        fsk_cnn_demod(input_I, input_Q, output, &predicted_bit);
        
        // Print results
        std::cout << "\nNetwork outputs:\n";
        std::cout << "  Class 0 (bit=0): " << output[0].to_float() << "\n";
        std::cout << "  Class 1 (bit=1): " << output[1].to_float() << "\n";
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
