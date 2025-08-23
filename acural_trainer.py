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

# Import our model factory
from models import get_model

# Suppress KMeans warning about n_init
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.cluster._kmeans")

def get_config():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Acural Recursion trainer.')
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['MNIST', 'CIFAR10'],
                        help='The dataset to use.')
    parser.add_argument('--model', type=str, default='resnet50',
                        help='The model architecture to use.')
    return parser.parse_args()

# --- Heuristic Filter Implementations (no changes needed) ---
def get_low_confidence_indices(model, dataset, device, entropy_threshold):
    model.eval()
    indices = []
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)
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
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)
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
    pool_loader = DataLoader(pool_dataset, batch_size=256, shuffle=False, num_workers=2)
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
    candidate_loader = DataLoader(candidate_dataset, batch_size=256, shuffle=False, num_workers=2)
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

def run_acural_recursion(args):
    """Main function to run the Acural Recursion training loop."""
    print(f"--- Starting Acural Recursion for {args.dataset} with model {args.model} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Dataset-specific Configurations ---
    configs = {
        'MNIST': {
            'model_name': 'MNIST_CNN',
            'dataset_loader': torchvision.datasets.MNIST,
            'transform': transforms.Compose([transforms.Resize(224), transforms.Grayscale(3), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
            'num_classes': 10, 'initial_pool_size': 500, 'epochs_per_iteration': 2, 'max_iterations': 5,
            'analysis_pool_size': 10000, 'entropy_threshold': 1.6, 'high_loss_k_percent': 5,
            'uniqueness_n_clusters': 20, 'uniqueness_similarity_threshold': 0.96,
            'baseline_accuracy': 98.97
        },
        'CIFAR10': {
            'model_name': args.model, # Use model from args
            'dataset_loader': torchvision.datasets.CIFAR10,
            'transform': transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            'num_classes': 10, 'initial_pool_size': 1000, 'epochs_per_iteration': 2, 'max_iterations': 4,
            'analysis_pool_size': 10000, 'entropy_threshold': 2.0, 'high_loss_k_percent': 10,
            'uniqueness_n_clusters': 50, 'uniqueness_similarity_threshold': 0.98,
            'baseline_accuracy': 71.42 # This will be updated after the new baseline run
        }
    }
    config = configs[args.dataset]

    # --- Data Loading ---
    full_train_dataset = config['dataset_loader'](root='./data', train=True, download=True, transform=config['transform'])
    test_dataset = config['dataset_loader'](root='./data', train=False, download=True, transform=config['transform'])
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    # --- Initialization ---
    model = get_model(config['model_name'], num_classes=config['num_classes'], pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')

    params_to_train = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_train, lr=0.001)

    all_indices = list(range(len(full_train_dataset)))
    np.random.seed(42)
    np.random.shuffle(all_indices)

    pool_indices = all_indices[:config['initial_pool_size']]
    candidate_indices = all_indices[config['initial_pool_size']:]
    print(f"Initial pool size: {len(pool_indices)}, Candidate pool size: {len(candidate_indices)}")

    # --- Main Loop ---
    for iteration in range(config['max_iterations']):
        print(f"\n--- Iteration {iteration + 1}/{config['max_iterations']} ---")
        pool_dataset = Subset(full_train_dataset, pool_indices)

        print(f"Training on current pool of {len(pool_indices)} samples...")
        model.train()
        # Ensure only the intended layers are trainable
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.requires_grad = True # This seems redundant, but ensures only head is trained
            else:
                param.requires_grad = False

        pool_loader = DataLoader(pool_dataset, batch_size=32, shuffle=True, num_workers=2)
        for _ in range(config['epochs_per_iteration']):
            for images, labels in pool_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                loss.backward()
                optimizer.step()

        if not candidate_indices: break

        analysis_indices = np.random.choice(candidate_indices, min(len(candidate_indices), config['analysis_pool_size']), replace=False).tolist()
        analysis_dataset = Subset(full_train_dataset, analysis_indices)
        print(f"Analyzing a random subset of {len(analysis_indices)} candidates...")

        low_conf_idx = get_low_confidence_indices(model, analysis_dataset, device, config['entropy_threshold'])
        high_loss_idx = get_high_loss_indices(model, analysis_dataset, device, criterion, config['high_loss_k_percent'])
        unique_idx = get_unique_indices(model, pool_dataset, analysis_dataset, device, config['uniqueness_n_clusters'], config['uniqueness_similarity_threshold'])

        new_indices = set(low_conf_idx) | set(high_loss_idx) | set(unique_idx)
        print(f"Selected {len(new_indices)} new data points.")

        if not new_indices: break

        pool_indices.extend(list(new_indices))
        candidate_indices = [idx for idx in candidate_indices if idx not in new_indices]

        accuracy = evaluate_model(model, test_loader, device)
        print(f"Iteration {iteration + 1} | Pool Size: {len(pool_indices)} | Test Accuracy: {accuracy:.2f}%")

    # --- Final Evaluation ---
    final_accuracy = evaluate_model(model, test_loader, device)
    print(f'\n==================================================')
    print(f'Final Test Accuracy for {args.model} on {args.dataset}: {final_accuracy:.2f}%')
    print(f'Using {len(pool_indices)} / {len(full_train_dataset)} training samples.')
    print(f"Baseline Accuracy was: {config['baseline_accuracy']}%")
    print(f'==================================================')

if __name__ == '__main__':
    args = get_config()
    run_acural_recursion(args)
