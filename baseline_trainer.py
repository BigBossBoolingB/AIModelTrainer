import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse

# Import our model factory
from models import get_model

def get_config():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Baseline model trainer for various datasets and models.')
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['MNIST', 'CIFAR10'],
                        help='The dataset to use.')
    parser.add_argument('--model', type=str, default='resnet50',
                        help='The model architecture to use (e.g., MNIST_CNN, CIFAR10_CNN, resnet50).')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training.')
    return parser.parse_args()

def train_and_evaluate(args):
    """
    Trains and evaluates a baseline model on the specified dataset.
    """
    print(f"--- Starting Baseline Training: Model={args.model}, Dataset={args.dataset} ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading and Transformation ---
    if args.dataset == 'MNIST':
        # MNIST specific transforms and data loader
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3), # ResNet expects 3 channels
            transforms.Resize(224), # ResNet expects 224x224 images
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset_loader = torchvision.datasets.MNIST
        num_classes = 10
    elif args.dataset == 'CIFAR10':
        # CIFAR-10 specific transforms and data loader
        transform = transforms.Compose([
            transforms.Resize(224), # ResNet expects 224x224 images
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset_loader = torchvision.datasets.CIFAR10
        num_classes = 10
    else:
        raise ValueError("Invalid dataset specified")

    train_dataset = dataset_loader(root='./data', train=True, download=True, transform=transform)
    test_dataset = dataset_loader(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print("Dataset loaded.")

    # --- Model Initialization ---
    # Use our factory to get the right model
    model = get_model(args.model, num_classes=num_classes, pretrained=True).to(device)
    print(f"Model {args.model} loaded.")

    # --- Optimizer Setup for Fine-Tuning ---
    # We only want to train the parameters of the final, newly-added layer.
    params_to_train = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_train.append(param)
            print(f"\tTraining parameter: {name}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params_to_train, lr=args.lr)

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
        print(f"Epoch [{epoch+1}/{args.epochs}] complete. Average Loss: {running_loss / len(train_loader):.4f}")

    # --- Evaluation Loop ---
    print("\nStarting evaluation...")
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'\n==================================================')
    print(f'Test Accuracy for {args.model} on {args.dataset}: {accuracy:.2f}%')
    print(f'==================================================')

if __name__ == '__main__':
    args = get_config()
    train_and_evaluate(args)
