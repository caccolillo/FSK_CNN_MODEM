import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Parameters
# -------------------------------------------------
fs = 8_000_000              # sample rate [Hz]
f0 = 800_000                # frequency for bit 0 [Hz]
f1 = 1_000_000              # frequency for bit 1 [Hz]

num_symbols = 10
samples_per_symbol = 8

# DDS / LUT parameters
PHASE_BITS = 10             # LUT size = 2^PHASE_BITS
LUT_SIZE = 2**PHASE_BITS
PHASE_MASK = LUT_SIZE - 1

# -------------------------------------------------
# Generate sine / cosine LUT
# -------------------------------------------------
n = np.arange(LUT_SIZE)
sin_lut = np.sin(2 * np.pi * n / LUT_SIZE)
cos_lut = np.cos(2 * np.pi * n / LUT_SIZE)

# -------------------------------------------------
# Generate random binary data
# -------------------------------------------------
bits = np.random.randint(0, 2, num_symbols)
print("Random binary sequence:", bits)

# -------------------------------------------------
# Map bits to frequency control words
# Δphase = f / fs * 2^PHASE_BITS
# -------------------------------------------------
fcw0 = int(f0 / fs * LUT_SIZE)
fcw1 = int(f1 / fs * LUT_SIZE)

fcw = np.repeat(np.where(bits == 0, fcw0, fcw1),
                samples_per_symbol)

# -------------------------------------------------
# DDS Phase Accumulator
# -------------------------------------------------
phase_acc = 0
phase_idx = np.zeros(len(fcw), dtype=int)

for i in range(len(fcw)):
    phase_acc = (phase_acc + fcw[i]) & PHASE_MASK
    phase_idx[i] = phase_acc

# -------------------------------------------------
# I / Q generation from LUT
# -------------------------------------------------
I = cos_lut[phase_idx]
Q = sin_lut[phase_idx]

# -------------------------------------------------
# Time vector (for plotting)
# -------------------------------------------------
t = np.arange(len(I)) / fs
data_plot = np.repeat(bits, samples_per_symbol)

# -------------------------------------------------
# Plot
# -------------------------------------------------
plt.figure(figsize=(8, 4))

plt.subplot(2, 1, 1)
plt.step(t * 1e6, data_plot, where="post")
plt.ylim(-0.2, 1.2)
plt.ylabel("Data")
plt.title("LUT-Based I/Q FSK Modulator (DDS Style)")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t * 1e6, I, label="I")
plt.plot(t * 1e6, Q, label="Q")
plt.xlabel("Time [µs]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

