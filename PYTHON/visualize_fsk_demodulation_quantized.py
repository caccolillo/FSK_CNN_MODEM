import numpy as np

# === 1. Load quantized parameters ===
quantized_params = np.load('quantized_params.npy', allow_pickle=True).item()
print("Loaded quantized parameters from 'quantized_params.npy'")

# === 2. Tiny FP5 CNN inference function ===
def cnn_fsk_fp5(symbol, quantized_params):
    conv_w = quantized_params['conv.weight']['quantized']  # (1,2,1)
    conv_b = quantized_params['conv.bias']['quantized']     # (1,)
    scale_w = quantized_params['conv.weight']['scale']
    scale_b = quantized_params['conv.bias']['scale']

    conv_out = np.zeros(8, dtype=np.float32)
    for t in range(8):
        conv_sum = 0
        for i in range(2):
            conv_sum += symbol[i, t] * conv_w[0, i, 0] * scale_w
        conv_sum += conv_b[0] * scale_b
        conv_out[t] = conv_sum

    # MaxPool1d
    pool_out = np.array([max(conv_out[0], conv_out[1]),
                         max(conv_out[2], conv_out[3]),
                         max(conv_out[4], conv_out[5]),
                         max(conv_out[6], conv_out[7])])

    # Fully connected
    fc_w = quantized_params['fc.weight']['quantized']  # (2,4)
    fc_b = quantized_params['fc.bias']['quantized']     # (2,)
    scale_w_fc = quantized_params['fc.weight']['scale']
    scale_b_fc = quantized_params['fc.bias']['scale']

    fc_out = np.zeros(2, dtype=np.float32)
    for j in range(2):
        s = 0
        for i in range(4):
            s += pool_out[i] * fc_w[j, i] * scale_w_fc
        s += fc_b[j] * scale_b_fc
        fc_out[j] = s

    # Softmax
    exp_out = np.exp(fc_out - np.max(fc_out))
    probs = exp_out / np.sum(exp_out)
    pred_bit = np.argmax(probs)

    return pred_bit, probs

# === 3. FSK Signal Parameters ===
SAMPLES_PER_SYMBOL = 8
NUM_SYMBOLS = 10

# Simple FSK symbol generator
def generate_fsk_symbol(bit):
    """Generate baseband I/Q symbol (without noise)"""
    phase_shift = 2 * np.pi * (1 if bit == 1 else -1) / SAMPLES_PER_SYMBOL
    I = np.cos(phase_shift * np.arange(SAMPLES_PER_SYMBOL))
    Q = np.sin(phase_shift * np.arange(SAMPLES_PER_SYMBOL))
    symbol = np.stack([I, Q], axis=0).astype(np.float32)
    return symbol

# === 4. Generate random bits and symbols ===
bits = np.random.randint(0, 2, NUM_SYMBOLS)
symbols = [generate_fsk_symbol(b) for b in bits]

# === 5. FP5 inference and comparison ===
correct = 0
print("\n=== FP5 Inference Results ===")
print(f"{'Symbol #':<8} {'Original':<8} {'Predicted':<10} {'Correct':<8} {'Probabilities'}")
print("-"*70)

for idx, (bit, sym) in enumerate(zip(bits, symbols)):
    pred, probs = cnn_fsk_fp5(sym, quantized_params)
    match = "✓" if pred == bit else "✗"
    if pred == bit:
        correct += 1
    print(f"{idx+1:<8} {bit:<8} {pred:<10} {match:<8} {probs}")

# === 6. Overall statistics ===
accuracy = 100 * correct / NUM_SYMBOLS
errors = NUM_SYMBOLS - correct
print("\n=== Overall Statistics ===")
print(f"Total symbols: {NUM_SYMBOLS}")
print(f"Correct predictions: {correct}")
print(f"Errors: {errors}")
print(f"Accuracy: {accuracy:.2f}%")

