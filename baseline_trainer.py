import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import yaml
import logging

from models import get_model
from utils import setup_logging

def load_config():
    """Loads the YAML configuration file."""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def build_transforms(transform_config):
    """Builds a torchvision.transforms.Compose object from a config list."""
    transform_list = []
    for t_config in transform_config:
        t_name = t_config['name']
        t_params = t_config.get('params', {})
        if hasattr(transforms, t_name):
            transform_list.append(getattr(transforms, t_name)(**t_params))
        else:
            raise ValueError(f"Transform {t_name} not recognized in torchvision.transforms")
    return transforms.Compose(transform_list)

def train_and_evaluate(config, dataset_name, model_name, logger):
    """
    Trains and evaluates a baseline model on the specified dataset using settings from the config.
    """
    logger.info(f"--- Starting Baseline Training: Model={model_name}, Dataset={dataset_name} ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Get configs ---
    train_cfg = config['training']
    dataset_cfg = config['datasets'][dataset_name]

    # --- Data Loading and Transformation ---
    transform = build_transforms(dataset_cfg['transform'])
    dataset_loader = getattr(torchvision.datasets, dataset_name)

    train_dataset = dataset_loader(root='./data', train=True, download=True, transform=transform)
    test_dataset = dataset_loader(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=train_cfg['batch_size'], shuffle=False, num_workers=2)
    logger.info("Dataset loaded.")

    # --- Model Initialization ---
    model = get_model(model_name, num_classes=dataset_cfg['num_classes'], pretrained=True).to(device)
    logger.info(f"Model {model_name} loaded.")

    # --- Optimizer Setup for Fine-Tuning ---
    params_to_train = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Found {len(params_to_train)} trainable parameters.")
    optimizer = optim.Adam(params_to_train, lr=train_cfg['lr'])
    criterion = nn.CrossEntropyLoss()

    # --- Training Loop ---
    logger.info(f"Starting training for {train_cfg['epochs']} epochs...")
    for epoch in range(train_cfg['epochs']):
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
        logger.info(f"Epoch [{epoch+1}/{train_cfg['epochs']}] complete. Average Loss: {running_loss / len(train_loader):.4f}")

    # --- Evaluation Loop ---
    logger.info("\nStarting evaluation...")
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
    logger.info(f'==================================================')
    logger.info(f'Test Accuracy for {model_name} on {dataset_name}: {accuracy:.2f}%')
    logger.info(f'==================================================')

if __name__ == '__main__':
    logger = setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset to use (e.g., CIFAR10). Overrides config.')
    parser.add_argument('--model', type=str, help='Model to use (e.g., resnet50). Overrides config.')
    cli_args = parser.parse_args()

    try:
        config = load_config()

        dataset_name = cli_args.dataset if cli_args.dataset else config['training']['default_dataset']
        model_name = cli_args.model if cli_args.model else config['training']['default_model']

        train_and_evaluate(config, dataset_name, model_name, logger)
    except Exception as e:
        logger.error(f"An error occurred during execution: {e}", exc_info=True)
        raise
