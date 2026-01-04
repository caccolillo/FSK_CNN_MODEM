import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# FSK Signal Parameters
F_CARRIER = 20e6  # 20 MHz
F_BITRATE = 1e6   # 1 Mbps
F_DEVIATION = 1e6  # 1 MHz
F_SAMPLING = 8e6  # 8 MHz
SAMPLES_PER_SYMBOL = int(F_SAMPLING / F_BITRATE)  # 8 samples

class CNNFSK(nn.Module):
    """Compact CNN for FSK demodulation - only 12 parameters"""
    
    def __init__(self):
        super(CNNFSK, self).__init__()
        
        # Convolutional layer: 1x1 kernel, 1 filter, stride=1
        self.conv = nn.Conv1d(in_channels=2, out_channels=1, 
                              kernel_size=1, stride=1, bias=True)
        
        # Max pooling: 2x2 window, stride=2
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Fully connected layer: 4 inputs, 2 outputs (bit 0 or 1)
        self.fc = nn.Linear(4, 2)
        
    def forward(self, x):
        # x shape: (batch, 2, 8)
        x = self.conv(x)  # (batch, 1, 8)
        x = self.maxpool(x)  # (batch, 1, 4)
        x = x.view(x.size(0), -1)  # (batch, 4)
        x = self.fc(x)  # (batch, 2)
        return x


def generate_fsk_symbol(bit, eb_n0_db):
    """Generate one FSK symbol (I/Q components) with AWGN noise"""
    # Frequency for bit 0: 19MHz, bit 1: 21MHz
    freq = F_CARRIER + (F_DEVIATION if bit == 1 else -F_DEVIATION)
    
    # Adjust for the actual aliased frequencies at baseband
    if bit == 0:
        phase_shift = 2 * np.pi * (-F_DEVIATION) / F_SAMPLING
    else:
        phase_shift = 2 * np.pi * F_DEVIATION / F_SAMPLING
        
    # Generate I and Q components
    I = np.cos(phase_shift * np.arange(SAMPLES_PER_SYMBOL))
    Q = np.sin(phase_shift * np.arange(SAMPLES_PER_SYMBOL))
    
    # Add AWGN noise
    eb_n0_linear = 10 ** (eb_n0_db / 10)
    noise_power = 1 / (2 * eb_n0_linear)
    noise_std = np.sqrt(noise_power)
    
    I += np.random.normal(0, noise_std, SAMPLES_PER_SYMBOL)
    Q += np.random.normal(0, noise_std, SAMPLES_PER_SYMBOL)
    
    # Stack I and Q into 2x8 array
    symbol = np.stack([I, Q], axis=0)
    
    return symbol.astype(np.float32), I, Q


def generate_bit_sequence(num_bits, eb_n0_db):
    """Generate a sequence of bits and their FSK symbols"""
    bits = np.random.randint(0, 2, num_bits)
    symbols = []
    I_sequence = []
    Q_sequence = []
    
    for bit in bits:
        symbol, I, Q = generate_fsk_symbol(bit, eb_n0_db)
        symbols.append(symbol)
        I_sequence.extend(I)
        Q_sequence.extend(Q)
    
    return bits, np.array(symbols), np.array(I_sequence), np.array(Q_sequence)


def demodulate_sequence(model, symbols, device):
    """Demodulate a sequence of FSK symbols using the CNN"""
    model.eval()
    demodulated_bits = []
    probabilities = []
    
    with torch.no_grad():
        # Convert to tensor
        symbols_tensor = torch.from_numpy(symbols).to(device)
        
        # Get predictions
        outputs = model(symbols_tensor)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        
        demodulated_bits = predicted.cpu().numpy()
        probabilities = probs.cpu().numpy()
    
    return demodulated_bits, probabilities


def plot_results(bits, I_sequence, Q_sequence, demodulated_bits, probabilities, eb_n0_db):
    """Plot the modulation, I/Q components, and demodulation results"""
    
    num_bits = len(bits)
    time_per_symbol = SAMPLES_PER_SYMBOL
    total_samples = len(I_sequence)
    time_axis = np.arange(total_samples) / F_SAMPLING * 1e6  # in microseconds
    
    # Create figure with subplots
    fig, axes = plt.subplots(5, 1, figsize=(14, 12))
    fig.suptitle(f'FSK Modulation and CNN Demodulation (Eb/N0 = {eb_n0_db} dB)', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Original Digital Sequence
    ax1 = axes[0]
    bit_times = np.arange(num_bits) * time_per_symbol / F_SAMPLING * 1e6
    ax1.step(bit_times, bits, where='post', linewidth=2, color='blue')
    ax1.set_ylabel('Bit Value', fontsize=11, fontweight='bold')
    ax1.set_title('Original Digital Sequence', fontsize=12)
    ax1.set_ylim([-0.2, 1.2])
    ax1.set_yticks([0, 1])
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, time_axis[-1]])
    
    # Plot 2: I Component (In-Phase)
    ax2 = axes[1]
    ax2.plot(time_axis, I_sequence, linewidth=1.5, color='red', label='I (In-Phase)')
    ax2.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
    ax2.set_title('I Component (In-Phase)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.set_xlim([0, time_axis[-1]])
    
    # Plot 3: Q Component (Quadrature)
    ax3 = axes[2]
    ax3.plot(time_axis, Q_sequence, linewidth=1.5, color='green', label='Q (Quadrature)')
    ax3.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
    ax3.set_title('Q Component (Quadrature)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')
    ax3.set_xlim([0, time_axis[-1]])
    
    # Plot 4: I/Q Constellation
    ax4 = axes[3]
    colors = ['blue' if b == 0 else 'red' for b in bits]
    for i in range(num_bits):
        start_idx = i * SAMPLES_PER_SYMBOL
        end_idx = start_idx + SAMPLES_PER_SYMBOL
        ax4.scatter(I_sequence[start_idx:end_idx], Q_sequence[start_idx:end_idx], 
                   c=colors[i], alpha=0.6, s=30, label=f'Bit {bits[i]}' if i < 2 else '')
    ax4.set_xlabel('I (In-Phase)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Q (Quadrature)', fontsize=11, fontweight='bold')
    ax4.set_title('I/Q Constellation Diagram', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend(['Bit 0', 'Bit 1'])
    ax4.axis('equal')
    
    # Plot 5: Demodulated Digital Sequence with Probabilities
    ax5 = axes[4]
    # Plot demodulated bits
    ax5_twin = ax5.twinx()
    ax5.step(bit_times, demodulated_bits, where='post', linewidth=2, 
             color='purple', label='Demodulated', alpha=0.8)
    ax5.step(bit_times, bits, where='post', linewidth=1.5, 
             color='blue', linestyle='--', label='Original', alpha=0.5)
    
    # Plot confidence (probability of predicted class)
    confidence = np.max(probabilities, axis=1)
    ax5_twin.bar(bit_times, confidence, width=time_per_symbol / F_SAMPLING * 1e6 * 0.8,
                 alpha=0.3, color='gray', label='Confidence')
    
    ax5.set_ylabel('Bit Value', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Time (μs)', fontsize=11, fontweight='bold')
    ax5.set_title('Demodulated Digital Sequence (CNN Output)', fontsize=12)
    ax5.set_ylim([-0.2, 1.2])
    ax5.set_yticks([0, 1])
    ax5_twin.set_ylabel('Confidence', fontsize=11, fontweight='bold')
    ax5_twin.set_ylim([0, 1.2])
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='upper left')
    ax5_twin.legend(loc='upper right')
    ax5.set_xlim([0, time_axis[-1]])
    
    plt.tight_layout()
    plt.savefig('fsk_demodulation_visualization.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'fsk_demodulation_visualization.png'")
    plt.show()


def print_comparison(bits, demodulated_bits, probabilities):
    """Print detailed comparison of original and demodulated bits"""
    print("\n" + "="*70)
    print("BIT-BY-BIT COMPARISON")
    print("="*70)
    print(f"{'Bit #':<8} {'Original':<12} {'Demodulated':<15} {'Prob[0]':<12} {'Prob[1]':<12} {'Match':<8}")
    print("-"*70)
    
    errors = 0
    for i in range(len(bits)):
        match = "✓" if bits[i] == demodulated_bits[i] else "✗"
        if bits[i] != demodulated_bits[i]:
            errors += 1
        
        print(f"{i:<8} {bits[i]:<12} {demodulated_bits[i]:<15} "
              f"{probabilities[i][0]:<12.4f} {probabilities[i][1]:<12.4f} {match:<8}")
    
    print("-"*70)
    accuracy = 100 * (1 - errors / len(bits))
    print(f"Total Bits: {len(bits)}")
    print(f"Errors: {errors}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("="*70)


def main():
    # Configuration
    NUM_BITS = 20  # Number of bits to generate
    EB_N0_DB = 10  # Signal-to-noise ratio in dB
    
    print("="*70)
    print("FSK CNN DEMODULATOR - VISUALIZATION SCRIPT")
    print("="*70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load the trained model
    print("\nLoading trained model...")
    model = CNNFSK().to(device)
    
    try:
        model.load_state_dict(torch.load('best_fsk_cnn.pth', map_location=device))
        print("✓ Model loaded successfully from 'best_fsk_cnn.pth'")
    except FileNotFoundError:
        print("✗ Error: 'best_fsk_cnn.pth' not found!")
        print("Please run the training script first.")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    # Print model parameters
    print(f"\nModel has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Generate random bit sequence with FSK modulation
    print(f"\nGenerating {NUM_BITS} random bits with FSK modulation...")
    print(f"Eb/N0 = {EB_N0_DB} dB")
    print(f"Carrier Frequency = {F_CARRIER/1e6} MHz")
    print(f"Bit 0 Frequency = {(F_CARRIER - F_DEVIATION)/1e6} MHz")
    print(f"Bit 1 Frequency = {(F_CARRIER + F_DEVIATION)/1e6} MHz")
    print(f"Sample Rate = {F_SAMPLING/1e6} MHz")
    print(f"Samples per Symbol = {SAMPLES_PER_SYMBOL}")
    
    bits, symbols, I_sequence, Q_sequence = generate_bit_sequence(NUM_BITS, EB_N0_DB)
    
    print(f"\nOriginal bit sequence:")
    print(''.join(map(str, bits)))
    
    # Demodulate using CNN
    print("\nDemodulating using CNN...")
    demodulated_bits, probabilities = demodulate_sequence(model, symbols, device)
    
    print(f"\nDemodulated bit sequence:")
    print(''.join(map(str, demodulated_bits)))
    
    # Print detailed comparison
    print_comparison(bits, demodulated_bits, probabilities)
    
    # Plot results
    print("\nGenerating visualization...")
    plot_results(bits, I_sequence, Q_sequence, demodulated_bits, probabilities, EB_N0_DB)
    
    print("\n✓ Visualization complete!")
    print("="*70)


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    main()
