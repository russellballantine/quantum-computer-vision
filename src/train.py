import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_mnist_loaders
from quantum_filter import QuanvolutionalFilter
from model import ClassicalPostProcessingCNN

# 1. Hardware Selection: Isolate Apple Silicon GPU Acceleration
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("--> Target Hardware: Apple Silicon GPU (MPS) selected for classical layers.")
else:
    device = torch.device("cpu")
    print("--> Target Hardware: MPS not available. Defaulting to CPU.")

# 2. Unified Hybrid QCNN Architecture Model Wrapper
class HybridQCNN(nn.Module):
    def __init__(self):
        super(HybridQCNN, self).__init__()
        # Quantum filter operates on CPU state-vectors
        self.quantum_filter = QuanvolutionalFilter(n_layers=1)
        # Classical post-processing network runs on target acceleration hardware
        self.classical_cnn = ClassicalPostProcessingCNN(num_classes=10)

    def forward(self, x):
        # Phase A: Run sliding quantum window loop (Executed on CPU)
        # Input: (batch_size, 1, 14, 14) -> Output: (batch_size, 4, 7, 7)
        q_features = self.quantum_filter(x)
        
        # Phase B: Bridge data securely from CPU RAM to Apple GPU Core registers
        q_features = q_features.to(device)
        
        # Phase C: Feed feature arrays into classical CNN layers
        logits = self.classical_cnn(q_features)
        return logits

def run_one_epoch_test():
    print("\n--- Initialising Hybrid QCNN 1-Epoch Validation Loop ---")
    
    # 3. Pull down a micro subset of data for quick portfolio testing
    # Large batches take a long time on local simulators; batch size 4 keeps it moving fast
    train_loader, _ = get_mnist_loaders(batch_size=4)
    
    # 4. Instantiate model assets and link components to hardware
    model = HybridQCNN()
    model.classical_cnn.to(device) # Force classical layers onto Mac GPU
    
    # 5. Define Optimization Functions
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    print("Beginning 1-Epoch validation execution. Running micro batches...")
    
    # Run a micro test check over the first few batches to verify pipeline communications
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Labels map straight to target GPU memory structures
        labels = labels.to(device)
        
        # Reset tracking gradients
        optimizer.zero_grad()
        
        # Forward Pass across entire multi-file pipeline
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward Pass (Autograd + Parameter-Shift Gradient calculations)
        loss.backward()
        optimizer.step()
        
        print(f"Batch [{batch_idx + 1}/5] | Loss Value: {loss.item():.4f}")
        
        # Cap execution at 5 micro-batches so your terminal test finishes inside a minute
        if batch_idx >= 4:
            break

    print("\nVerification Test Completed! Full hybrid model communications functional.")

if __name__ == "__main__":
    run_one_epoch_test()
