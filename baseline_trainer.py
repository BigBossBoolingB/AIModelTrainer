import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse

# Import our model definitions
from models import MNIST_CNN, CIFAR10_CNN

def get_config():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Baseline model trainer.')
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10'],
                        help='The dataset to use (MNIST or CIFAR10)')
    parser.add_argument('--epochs', type=int, default=4,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    return parser.parse_args()

def train_and_evaluate(args):
    """
    Trains and evaluates a baseline CNN on the specified dataset.
    """
    print(f"--- Starting Baseline Model Training for {args.dataset} ---")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading and Model Selection ---
    if args.dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        model = MNIST_CNN().to(device)
    elif args.dataset == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 3-channel normalization
        ])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        model = CIFAR10_CNN().to(device)
    else:
        raise ValueError("Invalid dataset specified")

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print("Dataset loaded.")

    # Initialize loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        print(f"Epoch {epoch+1} average loss: {running_loss / len(train_loader):.4f}")

    # --- Evaluation Loop ---
    print("\nStarting evaluation...")
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'\n==================================================')
    print(f'Test Accuracy for {args.dataset}: {accuracy:.2f}%')
    print(f'==================================================')
    print("--- Baseline Model Training Complete ---")

if __name__ == '__main__':
    args = get_config()
    train_and_evaluate(args)
