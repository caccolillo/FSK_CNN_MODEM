import torch
import torch.nn as nn
import numpy as np
import os
from pytorch_nndct.apis import torch_quantizer

# --- 1. Define Model Architecture ---
class CNNFSK(nn.Module):
    def __init__(self):
        super(CNNFSK, self).__init__()
        self.conv = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1, stride=1, bias=True)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = nn.Linear(4, 2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# --- 2. Calibration Data Generator ---
def get_calibration_samples(num_samples=32):
    samples = []
    eb_n0_db = 10
    f_sampling = 8e6
    f_deviation = 1e6
    samples_per_symbol = 8
    
    for _ in range(num_samples):
        bit = np.random.randint(0, 2)
        phase_shift = 2 * np.pi * (f_deviation if bit == 1 else -f_deviation) / f_sampling
        I = np.cos(phase_shift * np.arange(samples_per_symbol))
        Q = np.sin(phase_shift * np.arange(samples_per_symbol))
        
        noise_std = np.sqrt(1 / (2 * (10**(eb_n0_db/10))))
        I += np.random.normal(0, noise_std, samples_per_symbol)
        Q += np.random.normal(0, noise_std, samples_per_symbol)
        
        symbol = np.stack([I, Q], axis=0)
        samples.append(symbol)
        
    return torch.tensor(np.array(samples), dtype=torch.float32)

def run_quantization(model_path, quant_mode='calib'):
    device = torch.device("cpu")
    model = CNNFSK().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # CRITICAL FIX: 
    # Use batch size 32 for 'calib' to get better statistics.
    # Use batch size 1 for 'test' to satisfy the XModel export requirement.
    batch_size = 32 if quant_mode == 'calib' else 1
    inputs = get_calibration_samples(num_samples=batch_size)
    
    # Define dummy input shape for the quantizer (Batch size 1)
    dummy_input = torch.randn(1, 2, 8)

    # 3. Create Quantizer
    quantizer = torch_quantizer(quant_mode, model, (dummy_input), device=device)
    quant_model = quantizer.quant_model

    # 4. Forward pass to trace the model
    with torch.no_grad():
        _ = quant_model(inputs)

    # 5. Handle output
    if quant_mode == 'calib':
        quantizer.export_quant_config()
        print("\n[SUCCESS] Calibration finished. Config saved to ./quantize_result/")
    else:
        # Export the final xmodel for the DPU
        quantizer.export_xmodel(deploy_check=False, output_dir="quantize_result")
        print("\n[SUCCESS] Quantized xmodel saved to ./quantize_result/CNNFSK_int.xmodel")

if __name__ == "__main__":
    MODEL_FILE = 'best_fsk_cnn.pth'
    
    if not os.path.exists(MODEL_FILE):
        print(f"Error: {MODEL_FILE} not found!")
    else:
        # Step 1: Calibration (Gathers statistics)
        print("\n" + "="*50)
        print("STEP 1: RUNNING CALIBRATION")
        print("="*50)
        run_quantization(MODEL_FILE, quant_mode='calib')
        
        # Step 2: Export (Generates hardware-compatible files)
        print("\n" + "="*50)
        print("STEP 2: EXPORTING XMODEL")
        print("="*50)
        run_quantization(MODEL_FILE, quant_mode='test')
