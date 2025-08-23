import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import entropy
import warnings
import argparse
import yaml
import logging

from models import get_model
from utils import setup_logging

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.cluster._kmeans")

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def build_transforms(transform_config):
    transform_list = []
    for t_config in transform_config:
        t_name = t_config['name']
        t_params = t_config.get('params', {})
        if hasattr(transforms, t_name):
            transform_list.append(getattr(transforms, t_name)(**t_params))
        else:
            raise ValueError(f"Transform {t_name} not recognized")
    return transforms.Compose(transform_list)

# --- Heuristic Filter Implementations (no changes needed) ---
def get_low_confidence_indices(model, dataset, device, entropy_threshold):
    model.eval()
    indices = []
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=2)
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            batch_entropy = entropy(probs.cpu().numpy(), axis=1)
            start_idx = i * loader.batch_size
            original_batch_indices = dataset.indices[start_idx : start_idx + len(images)]
            for j, ent in enumerate(batch_entropy):
                if ent > entropy_threshold:
                    indices.append(original_batch_indices[j])
    return indices

def get_high_loss_indices(model, dataset, device, criterion, k_percent):
    model.eval()
    losses, original_indices_map = [], []
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=2)
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            batch_losses = [criterion(output.unsqueeze(0), label.unsqueeze(0)).item() for output, label in zip(outputs, labels)]
            losses.extend(batch_losses)
            start_idx = i * loader.batch_size
            original_indices_map.extend(dataset.indices[start_idx : start_idx + len(images)])
    sorted_indices = np.argsort(losses)[::-1]
    num_to_select = int(len(losses) * (k_percent / 100.0))
    return [original_indices_map[i] for i in sorted_indices[:num_to_select]]

def get_unique_indices(model, pool_dataset, candidate_dataset, device, n_clusters, similarity_threshold):
    model.eval()
    pool_features_list = []
    pool_loader = DataLoader(pool_dataset, batch_size=512, shuffle=False, num_workers=2)
    with torch.no_grad():
        for images, _ in pool_loader:
            pool_features_list.append(model.get_features(images).cpu().numpy())
    if not pool_features_list: return candidate_dataset.indices
    pool_features = np.vstack(pool_features_list)
    if len(pool_features) < n_clusters:
        centroids = torch.tensor(pool_features).to(device)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(pool_features)
        centroids = torch.tensor(kmeans.cluster_centers_).to(device)

    indices = []
    candidate_loader = DataLoader(candidate_dataset, batch_size=512, shuffle=False, num_workers=2)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    with torch.no_grad():
        for i, (images, _) in enumerate(candidate_loader):
            candidate_features = model.get_features(images.to(device))
            max_similarity, _ = torch.max(cos(candidate_features.unsqueeze(1), centroids.unsqueeze(0)), dim=1)
            start_idx = i * candidate_loader.batch_size
            original_batch_indices = candidate_dataset.indices[start_idx : start_idx + len(images)]
            for j, sim in enumerate(max_similarity):
                if sim < similarity_threshold:
                    indices.append(original_batch_indices[j])
    return indices

def evaluate_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def run_acural_recursion(config, dataset_name, model_name, logger):
    """Main function to run the Acural Recursion training loop."""
    logger.info(f"--- Starting Acural Recursion for {dataset_name} with model {model_name} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Get configs ---
    train_cfg = config['training']
    dataset_cfg = config['datasets'][dataset_name]
    acural_cfg = config['acural_recursion'][dataset_name]

    # --- Data Loading ---
    transform = build_transforms(dataset_cfg['transform'])
    dataset_loader = getattr(torchvision.datasets, dataset_name)
    full_train_dataset = dataset_loader(root='./data', train=True, download=True, transform=transform)
    test_dataset = dataset_loader(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=train_cfg['batch_size'], shuffle=False, num_workers=2)

    # --- Initialization ---
    model = get_model(model_name, num_classes=dataset_cfg['num_classes'], pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=train_cfg['lr'])

    all_indices = list(range(len(full_train_dataset)))
    np.random.seed(42)
    np.random.shuffle(all_indices)

    pool_indices = all_indices[:acural_cfg['initial_pool_size']]
    candidate_indices = all_indices[acural_cfg['initial_pool_size']:]
    logger.info(f"Initial pool size: {len(pool_indices)}, Candidate pool size: {len(candidate_indices)}")

    # --- Main Loop ---
    for iteration in range(acural_cfg['max_iterations']):
        logger.info(f"\n--- Iteration {iteration + 1}/{acural_cfg['max_iterations']} ---")
        pool_dataset = Subset(full_train_dataset, pool_indices)

        logger.info(f"Training on current pool of {len(pool_indices)} samples...")
        model.train()
        pool_loader = DataLoader(pool_dataset, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=2)
        for _ in range(acural_cfg['epochs_per_iteration']):
            for images, labels in pool_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                loss.backward()
                optimizer.step()

        if not candidate_indices:
            logger.info("Candidate pool is empty. Finalizing training.")
            break

        analysis_indices = np.random.choice(candidate_indices, min(len(candidate_indices), acural_cfg['analysis_pool_size']), replace=False).tolist()
        analysis_dataset = Subset(full_train_dataset, analysis_indices)
        logger.info(f"Analyzing a random subset of {len(analysis_indices)} candidates...")

        low_conf_idx = get_low_confidence_indices(model, analysis_dataset, device, acural_cfg['entropy_threshold'])
        high_loss_idx = get_high_loss_indices(model, analysis_dataset, device, criterion, acural_cfg['high_loss_k_percent'])
        unique_idx = get_unique_indices(model, pool_dataset, analysis_dataset, device, acural_cfg['uniqueness_n_clusters'], acural_cfg['uniqueness_similarity_threshold'])

        new_indices = set(low_conf_idx) | set(high_loss_idx) | set(unique_idx)
        logger.info(f"Selected {len(new_indices)} new data points.")

        if not new_indices:
            logger.info("No new data points selected. Ending training early.")
            break

        pool_indices.extend(list(new_indices))
        candidate_indices = [idx for idx in candidate_indices if idx not in new_indices]

        accuracy = evaluate_model(model, test_loader, device)
        logger.info(f"Iteration {iteration + 1} | Pool Size: {len(pool_indices)} | Test Accuracy: {accuracy:.2f}%")

    # --- Final Evaluation ---
    final_accuracy = evaluate_model(model, test_loader, device)
    logger.info(f'\n==================================================')
    logger.info(f'Final Test Accuracy for {model_name} on {dataset_name}: {final_accuracy:.2f}%')
    logger.info(f'Using {len(pool_indices)} / {len(full_train_dataset)} training samples.')
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

        run_acural_recursion(config, dataset_name, model_name, logger)
    except Exception as e:
        logger.error(f"An error occurred during execution: {e}", exc_info=True)
        raise
