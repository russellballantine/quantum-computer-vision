import torch
import torch.nn as nn

class ClassicalPostProcessingCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ClassicalPostProcessingCNN, self).__init__()
        
        # Inbound shape: (batch_size, 4, 7, 7) from the upcoming 4-qubit quantum filter
        self.conv_block = nn.Sequential(
            # Expand 4 quantum channels into 16 classical feature channels
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # Further extract spatial features
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Adaptive pooling forces the spatial size down to 1x1 regardless of input size
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final classification head mapping 32 features to 10 digit classes
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        # 1. Forward pass through convolutional features
        x = self.conv_block(x)
        
        # 2. Pool down to (batch_size, 32, 1, 1)
        x = self.global_pool(x)
        
        # 3. Flatten the tensor to (batch_size, 32)
        x = torch.flatten(x, start_dim=1)
        
        # 4. Output logit predictions for CrossEntropyLoss
        logits = self.fc(x)
        return logits

# Structural verification loop to test matrix dimensions locally
if __name__ == "__main__":
    # Simulate a batch of 32 images processed through a 4-qubit 2x2 sliding window
    mock_quantum_features = torch.randn(32, 4, 7, 7)
    
    # Initialize model
    model = ClassicalPostProcessingCNN(num_classes=10)
    
    # Forward pass
    predictions = model(mock_quantum_features)
    
    print("--- Classical CNN Structural Verification ---")
    print(f"Input shape from quantum filter: {mock_quantum_features.shape}")
    print(f"Output classification shape:     {predictions.shape}")
    print("Classical CNN structural verification test passed successfully!")
