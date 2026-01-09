# FSK_CNN_MODEM

**CNN‑Based FSK Demodulator With FPGA Implementation**

This repository contains a complete workflow for designing, training, and deploying a **Convolutional Neural Network (CNN)**‑based Frequency‑Shift Keying (FSK) modem/demodulator, including FPGA acceleration using High‑Level Synthesis (HLS) and Vitis AI.

The project combines **signal processing, machine learning, and FPGA design** to demonstrate how a neural network can be trained to demodulate FSK symbols and then deployed on embedded hardware.

---

## Overview

Frequency‑Shift Keying (FSK) is a digital modulation technique where information is encoded by shifting the carrier frequency between discrete values. Traditional FSK demodulators rely on DSP algorithms (filters, PLLs, discriminators).

In this project, a **CNN learns to directly classify FSK symbols from I/Q samples**, enabling:
- Robust demodulation under noise and impairments
- Hardware‑accelerated inference on FPGA
- A full ML‑to‑hardware workflow

---

## Repository Structure

```
FSK_CNN_MODEM/
├── DOCS/                   # Documentation and design notes
├── HLS/                    # HLS CNN accelerator implementation
├── PYTHON/                 # Dataset generation, training, inference
├── VITIS_AI/               # Vitis AI quantization & deployment
├── VIVADO/FSK_CNN_DEMOD/   # Vivado block design and integration
├── README.md
└── README.pdf
```

---

## Features

### FSK Signal Processing
- Synthetic FSK signal generation
- Configurable sampling rate, deviation, and SNR
- I/Q‑based symbol representation

### CNN‑Based Demodulation
- CNN architecture optimized for short symbol windows
- Supervised training pipeline
- Bit‑level classification output

### FPGA Deployment
- HLS implementation of CNN inference
- Fixed‑point / quantized arithmetic support
- Vitis AI integration for DPU‑based execution
- Vivado hardware wrapper

---

## Dependencies

### Python
- Python 3.8+
- numpy
- scipy
- torch or tensorflow (depending on training flow)

### FPGA Toolchain
- Xilinx Vivado
- Vitis
- Vitis AI
- Supported FPGA platform (e.g. Zynq UltraScale+)

> Tool versions depend on the target platform and are documented in `DOCS/`.

---

## Workflow

###  Dataset Generation
Generate synthetic FSK symbols with controlled noise and channel conditions.

```bash
cd PYTHON
python generate_dataset.py
```

###  CNN Training
Train the CNN to classify FSK symbols.

```bash
python train_model.py
```

###  Inference & Evaluation
Evaluate BER and classification accuracy.

```bash
python inference.py
```

###  FPGA Deployment
1. Quantize the trained model using Vitis AI 
2. Generate HLS CNN accelerator 
3. Build Vivado design 
4. Deploy on target FPGA 

---

##  Metrics

- Classification accuracy
- Bit Error Rate (BER)
- FPGA latency and throughput

---

