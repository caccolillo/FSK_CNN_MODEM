import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# FSK Signal Parameters
F_CARRIER = 20e6  # 20 MHz
F_BITRATE = 1e6   # 1 Mbps
F_DEVIATION = 1e6  # 1 MHz
F_SAMPLING = 8e6  # 8 MHz
SAMPLES_PER_SYMBOL = int(F_SAMPLING / F_BITRATE)  # 8 samples

# Training Parameters
TRAIN_SAMPLES = 200000  # 2e5
VAL_SAMPLES = 50000     # 5e4
BATCH_SIZE = 256
EPOCHS = 50

class FSKDataset(Dataset):
    """Generate FSK symbols with AWGN noise"""
    
    def __init__(self, num_samples, eb_n0_range=(0, 25)):
        self.num_samples = num_samples
        self.eb_n0_min, self.eb_n0_max = eb_n0_range
        
    def __len__(self):
        return self.num_samples
    
    def generate_fsk_symbol(self, bit, eb_n0_db):
        """Generate one FSK symbol (I/Q components)"""
        # Frequency for bit 0: 19MHz, bit 1: 21MHz
        freq = F_CARRIER + (F_DEVIATION if bit == 1 else -F_DEVIATION)
        
        # Time vector for one symbol
        t = np.arange(SAMPLES_PER_SYMBOL) / F_SAMPLING
        
        # Generate I and Q components (baseband after aliasing)
        # Due to undersampling, the signal appears at baseband
        baseband_freq = freq % F_SAMPLING
        if baseband_freq > F_SAMPLING / 2:
            baseband_freq = F_SAMPLING - baseband_freq
        
        # Adjust for the actual aliased frequencies
        if bit == 0:
            phase_shift = 2 * np.pi * (-F_DEVIATION) / F_SAMPLING
        else:
            phase_shift = 2 * np.pi * F_DEVIATION / F_SAMPLING
            
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
        
        return symbol.astype(np.float32)
    
    def __getitem__(self, idx):
        # Random bit (0 or 1)
        bit = np.random.randint(0, 2)
        
        # Random Eb/N0 from range
        eb_n0_db = np.random.uniform(self.eb_n0_min, self.eb_n0_max)
        
        # Generate symbol
        symbol = self.generate_fsk_symbol(bit, eb_n0_db)
        
        return torch.from_numpy(symbol), torch.tensor(bit, dtype=torch.long)


class CNNFSK(nn.Module):
    """Compact CNN for FSK demodulation - only 12 parameters"""
    
    def __init__(self):
        super(CNNFSK, self).__init__()
        
        # Convolutional layer: 1x1 kernel, 1 filter, stride=1
        # Input: (batch, 2, 8) -> Output: (batch, 1, 8)
        self.conv = nn.Conv1d(in_channels=2, out_channels=1, 
                              kernel_size=1, stride=1, bias=True)
        
        # Max pooling: 2x2 window, stride=2
        # Output: (batch, 1, 4)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Fully connected layer: 4 inputs, 2 outputs (bit 0 or 1)
        # 4*2 weights + 2 biases = 10 parameters
        self.fc = nn.Linear(4, 2)
        
    def forward(self, x):
        # x shape: (batch, 2, 8)
        
        # Convolutional layer
        x = self.conv(x)  # (batch, 1, 8)
        
        # Max pooling
        x = self.maxpool(x)  # (batch, 1, 4)
        
        # Flatten
        x = x.view(x.size(0), -1)  # (batch, 4)
        
        # Fully connected
        x = self.fc(x)  # (batch, 2)
        
        return x
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def save_float_parameters(model, filename='float_params.npz'):
    """Save floating point parameters"""
    float_params = {}
    
    for name, param in model.state_dict().items():
        float_params[name] = param.cpu().numpy()
    
    np.savez(filename, **float_params)
    print(f"Float parameters saved to '{filename}'")
    
    # Also save in readable text format
    txt_filename = filename.replace('.npz', '.txt')
    with open(txt_filename, 'w') as f:
        f.write("=== Float32 Parameters ===\n\n")
        for name, param in float_params.items():
            f.write(f"{name}:\n")
            f.write(f"  Shape: {param.shape}\n")
            f.write(f"  Values: {param}\n\n")
    
    print(f"Float parameters (text) saved to '{txt_filename}'")
    return float_params


def quantize_model(model, num_bits=5):
    """Quantize model parameters to fixed-point (FP5)"""
    quantized_state = {}
    
    for name, param in model.state_dict().items():
        # Convert to numpy
        param_np = param.cpu().numpy()
        
        # Find min/max for quantization range
        param_min = param_np.min()
        param_max = param_np.max()
        
        # Quantize to num_bits signed representation
        n_levels = 2 ** (num_bits - 1) - 1  # -15 to 15 for 5-bit signed
        
        # Scale to quantization range
        param_scaled = (param_np - param_min) / (param_max - param_min)
        param_quant = np.round(param_scaled * 2 * n_levels - n_levels)
        
        # Store quantization parameters
        quantized_state[name] = {
            'quantized': param_quant.astype(np.int8),
            'min': param_min,
            'max': param_max,
            'scale': (param_max - param_min) / (2 * n_levels)
        }
    
    return quantized_state


def save_quantized_parameters(quantized_params, filename='quantized_params.npz'):
    """Save quantized parameters"""
    # Save as numpy file
    np.save(filename.replace('.npz', '.npy'), quantized_params)
    print(f"Quantized parameters saved to '{filename.replace('.npz', '.npy')}'")
    
    # Also save in readable text format for VHDL implementation
    txt_filename = filename.replace('.npz', '.txt')
    with open(txt_filename, 'w') as f:
        f.write("=== Quantized Parameters (FP5) for FPGA Implementation ===\n\n")
        for name, data in quantized_params.items():
            f.write(f"{name}:\n")
            f.write(f"  Quantized values (int8): {data['quantized']}\n")
            f.write(f"  Min: {data['min']:.6f}\n")
            f.write(f"  Max: {data['max']:.6f}\n")
            f.write(f"  Scale: {data['scale']:.6f}\n\n")
    
    print(f"Quantized parameters (text) saved to '{txt_filename}'")


def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = FSKDataset(TRAIN_SAMPLES, eb_n0_range=(0, 25))
    val_dataset = FSKDataset(VAL_SAMPLES, eb_n0_range=(0, 25))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                          shuffle=False, num_workers=4)
    
    # Create model
    model = CNNFSK().to(device)
    print(f"\nModel created with {model.count_parameters()} parameters")
    print(f"Expected: 12 parameters (1 conv weight + 1 conv bias + 8 fc weights + 2 fc biases)")
    
    # Loss and optimizer (NAdam as specified in paper)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.NAdam(model.parameters(), 
                           lr=0.001,
                           betas=(0.9, 0.999),
                           eps=1e-7)
    
    # Training loop
    print("\nStarting training...")
    best_val_acc = 0
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                           optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_fsk_cnn.pth')
            print(f"  -> Saved best model with validation accuracy: {val_acc:.2f}%")
    
    print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.2f}%")
    
    # Load best model for parameter extraction
    model.load_state_dict(torch.load('best_fsk_cnn.pth'))
    
    # Save float parameters (non-quantized)
    print("\n" + "="*60)
    print("Saving float32 parameters...")
    float_params = save_float_parameters(model, 'float_params.npz')
    
    # Quantize model for FPGA implementation
    print("\n" + "="*60)
    print("Quantizing model to FP5...")
    quantized_params = quantize_model(model, num_bits=5)
    save_quantized_parameters(quantized_params, 'quantized_params.npz')
    
    # Print summary
    print("\n" + "="*60)
    print("=== Parameter Summary ===")
    print(f"\nFloat32 Parameters:")
    for name, param in float_params.items():
        print(f"  {name}: shape={param.shape}, range=[{param.min():.6f}, {param.max():.6f}]")
    
    print(f"\nQuantized (FP5) Parameters:")
    for name, data in quantized_params.items():
        print(f"  {name}: quantized_range=[{data['quantized'].min()}, {data['quantized'].max()}]")
    
    print("\n" + "="*60)
    print("Files saved:")
    print("  1. best_fsk_cnn.pth - PyTorch model weights")
    print("  2. float_params.npz - Float32 parameters (numpy)")
    print("  3. float_params.txt - Float32 parameters (readable)")
    print("  4. quantized_params.npy - Quantized FP5 parameters")
    print("  5. quantized_params.txt - Quantized FP5 parameters (readable)")
    print("="*60)


if __name__ == "__main__":
    main()
