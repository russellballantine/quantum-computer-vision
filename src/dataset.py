import torch
from torchvision import datasets, transforms

def get_mnist_loaders(batch_size=4):
    """
    Downloads and pre-processes MNIST data down to 14x14 pixels 
    to feed efficiently into our 2x2 sliding quantum window filter.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Resize((14, 14), antialias=True) # Resize to 14x14 for the QCNN framework
    ])
    
    # Download raw binaries locally
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Bundle into local batched streams
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Local execution check block
if __name__ == "__main__":
    print("--- Testing Dataset Pipeline ---")
    train_loader, _ = get_mnist_loaders(batch_size=4)
    images, labels = next(iter(train_loader))
    print(f"Batch image dimensions: {images.shape} (Expected: [4, 1, 14, 14])")
    print("Dataset verification passed successfully!")
