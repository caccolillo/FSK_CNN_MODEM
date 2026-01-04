import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
num_symbols = 5
samples_per_symbol = 8
fs = 8000000          # sample rate [Hz]
f0 = 800000          # frequency for bit 0 [Hz]
f1 = 1000000          # frequency for bit 1 [Hz]

# -----------------------------
# Generate random binary data
# -----------------------------
bits = np.random.randint(0, 2, num_symbols)
print("Random binary sequence:", bits)

# -----------------------------
# Time vector
# -----------------------------
t = np.arange(num_symbols * samples_per_symbol) / fs

# -----------------------------
# Map bits to frequencies
# -----------------------------
freqs = np.repeat(np.where(bits == 0, f0, f1),
                   samples_per_symbol)

# -----------------------------
# Phase accumulator (CPFSK)
# -----------------------------
phase = 2 * np.pi * np.cumsum(freqs) / fs

# -----------------------------
# I/Q generation
# -----------------------------
I = np.cos(phase)
Q = np.sin(phase)

# -----------------------------
# Plot
# -----------------------------
data_plot = np.repeat(bits, samples_per_symbol)

plt.figure(figsize=(8, 4))

plt.subplot(2, 1, 1)
plt.step(t, data_plot, where="post")
plt.ylim(-0.2, 1.2)
plt.ylabel("Data")
plt.title("Binary Data and I/Q FSK Signal")

plt.subplot(2, 1, 2)
plt.plot(t, I, label="I")
plt.plot(t, Q, label="Q")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()
plt.show()

