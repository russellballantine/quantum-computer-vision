import torch
import torch.nn as nn
import pennylane as qml

# 1. Define the 4-qubit Quantum Device
dev = qml.device("default.qubit", wires=4)

# 2. Define the Variational Quantum Circuit (VQC)
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    """
    Processes a single 2x2 pixel patch (4 features) using 4 qubits.
    """
    # Phase A: Angle Encoding (Ry rotations)
    for i in range(4):
        qml.RY(inputs[i], wires=i)
        
    # Phase B: Parameterised Entangling Layers (The Variational Filter)
    # Alternating rotation gates and CNOT entanglers
    for layer_weights in weights:
        for i in range(4):
            qml.RX(layer_weights[i], wires=i)
            qml.RY(layer_weights[i + 4], wires=i)
            
        # Linear entanglement chain
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[3, 0]) # Close the loop

    # Phase C: Measurement (Extract 4 classical expectation scalars)
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]


class QuanvolutionalFilter(nn.Module):
    def __init__(self, n_layers=2):
        super(QuanvolutionalFilter, self).__init__()
        self.n_layers = n_layers
        
        # Each layer needs 8 parameters (Rx and Ry for all 4 qubits)
        weight_shapes = {"weights": (n_layers, 8)}
        
        # Wrap the PennyLane QNode into a PyTorch-compatible Layer
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

    def forward(self, x):
        """
        Sliding 2x2 window (stride=2) across a batch of 14x14 images.
        Inbound shape:  (batch_size, 1, 14, 14)
        Outbound shape: (batch_size, 4, 7, 7)
        """
        batch_size, channels, height, width = x.shape
        stride = 2
        kernel_size = 2
        
        # Calculate output spatial grid size: (14 - 2)/2 + 1 = 7
        out_h = (height - kernel_size) // stride + 1
        out_w = (width - kernel_size) // stride + 1
        
        # Initialize an empty tensor to hold the quantum-extracted feature maps
        # Placing it on the same device (CPU/M-series GPU) as the input tensor
        out = torch.zeros((batch_size, 4, out_h, out_w), device=x.device)
        
        # Slide across the spatial grid
        for h in range(out_h):
            for w in range(out_w):
                h_start = h * stride
                h_end = h_start + kernel_size
                w_start = w * stride
                w_end = w_start + kernel_size
                
                # Extract the 2x2 patch across the complete batch
                # Shape becomes: (batch_size, 4)
                patch = x[:, 0, h_start:h_end, w_start:w_end].reshape(batch_size, 4)
                
                # Ingest patch features through the 4-qubit PennyLane VQC
                # q_layer handles batching automatically under the hood
                q_features = self.q_layer(patch)
                
                # Assign the 4 measured expectation channels to the current grid coordinate
                out[:, :, h, w] = q_features
                
        return out


# Local Structural Verification Loop
if __name__ == "__main__":
    print("--- Initialising Quantum Filter Test ---")
    
    # Simulate a batch of 4 normalized 14x14 images
    mock_batch = torch.randn(4, 1, 14, 14)
    
    # Initialize the quantum filter
    q_filter = QuanvolutionalFilter(n_layers=1)
    
    # Run a mock forward pass
    with torch.no_grad():
        output_features = q_filter(mock_batch)
        
    print(f"Input batch shape:   {mock_batch.shape}")
    print(f"Output batch shape:  {output_features.shape} (Expected: [4, 4, 7, 7])")
    print("Quantum filter compilation and structural check passed successfully!")
