import torch
from torchvision import datasets, transforms

def get_mnist_loaders(batch_size=32):
    """
    Downloads and prepares MNIST data resized to 14x14 pixels.
    Returns PyTorch DataLoaders for training and testing.
    """
    # Pipeline to transform raw data: convert to tensors, normalize, and resize down to 14x14
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Resize((14, 14), antialias=True)
    ])
    
    # Download MNIST dataset splits
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # DataLoaders handle shuffling and mini-batching for our PyTorch training loop
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

if __name__ == '__main__':
    # Verify the setup works locally
    train_loader, test_loader = get_mnist_loaders(batch_size=4)
    images, labels = next(iter(train_loader))
    print(f"Data loading successful!")
    print(f"Batch image shape (Expected: [4, 1, 14, 14]): {images.shape}")
    print(f"Batch labels: {labels}")
