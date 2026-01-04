#!/bin/bash

echo "=== FSK CNN Training Environment Setup ==="
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv fsk_cnn_env

# Activate virtual environment
echo "Activating virtual environment..."
source fsk_cnn_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch 1.11 with CUDA 11.3
echo "Installing PyTorch 1.11 with CUDA 11.3..."
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Install numpy
echo "Installing numpy..."
pip install numpy

# Verify installation
echo ""
echo "=== Verifying Installation ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=== Setup Complete! ==="
echo "Virtual environment is activated."
echo "You can now run: python train_fsk_cnn.py"
echo ""
echo "To deactivate later, run: deactivate"
echo "To reactivate, run: source fsk_cnn_env/bin/activate"
