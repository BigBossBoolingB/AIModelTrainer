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

# Suppress KMeans warning about n_init
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.cluster._kmeans")

# --- Model Definition (reusing from baseline for fair comparison) ---
class AcuralCNN(nn.Module):
    def __init__(self):
        super(AcuralCNN, self).__init__()
        # Feature extractor layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Classifier layers
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 64 * 7 * 7) # Flatten
        x = self.fc_layers(x)
        return x

    def get_features(self, x):
        """Extracts features from the convolutional layers for uniqueness analysis."""
        return self.conv_layers(x).view(-1, 64 * 7 * 7)

# --- Heuristic Filter Implementations ---

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
    losses = []
    original_indices_map = []
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
    top_k_indices = [original_indices_map[i] for i in sorted_indices[:num_to_select]]
    return top_k_indices

def get_unique_indices(model, pool_dataset, candidate_dataset, device, n_clusters, similarity_threshold):
    model.eval()
    indices = []

    pool_features_list = []
    pool_loader = DataLoader(pool_dataset, batch_size=512, shuffle=False, num_workers=2)
    with torch.no_grad():
        for images, _ in pool_loader:
            images = images.to(device)
            features = model.get_features(images)
            pool_features_list.append(features.cpu().numpy())

    if not pool_features_list:
        return candidate_dataset.indices

    pool_features = np.vstack(pool_features_list)

    if len(pool_features) < n_clusters:
        centroids = torch.tensor(pool_features).to(device)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(pool_features)
        centroids = torch.tensor(kmeans.cluster_centers_).to(device)

    candidate_loader = DataLoader(candidate_dataset, batch_size=512, shuffle=False, num_workers=2)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    with torch.no_grad():
        for i, (images, _) in enumerate(candidate_loader):
            images = images.to(device)
            candidate_features = model.get_features(images)
            similarity_matrix = cos(candidate_features.unsqueeze(1), centroids.unsqueeze(0))
            max_similarity, _ = torch.max(similarity_matrix, dim=1)

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

# --- Main Acural Recursion Training ---
def run_acural_recursion_poc():
    print("--- Starting Acural Recursion PoC ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters (tuned for faster execution in limited environment)
    initial_pool_size = 500
    batch_size = 32
    learning_rate = 0.001
    epochs_per_iteration = 2  # Reduced
    max_iterations = 5      # Reduced
    analysis_pool_size = 10000 # Analyze a subset of candidates
    entropy_threshold = 1.6
    high_loss_k_percent = 5
    uniqueness_n_clusters = 20
    uniqueness_similarity_threshold = 0.96

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    full_train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=2)

    model = AcuralCNN().to(device)
    criterion = nn.CrossEntropyLoss(reduction='none') # Use 'none' for individual losses
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    all_indices = list(range(len(full_train_dataset)))
    np.random.seed(42)
    np.random.shuffle(all_indices)

    pool_indices = all_indices[:initial_pool_size]
    candidate_indices = all_indices[initial_pool_size:]

    print(f"Initial pool size: {len(pool_indices)}, Candidate pool size: {len(candidate_indices)}")

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
        pool_dataset = Subset(full_train_dataset, pool_indices)

        print(f"Training on current pool of {len(pool_indices)} samples...")
        model.train()
        pool_loader = DataLoader(pool_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        for _ in range(epochs_per_iteration):
            for images, labels in pool_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = nn.CrossEntropyLoss()(outputs, labels) # Use default reduction for optimizer step
                loss.backward()
                optimizer.step()

        if not candidate_indices:
            print("Candidate pool is empty. Finalizing training.")
            break

        # Analyze a smaller, random subset of the candidate pool to save time
        if len(candidate_indices) > analysis_pool_size:
            analysis_indices = np.random.choice(candidate_indices, analysis_pool_size, replace=False).tolist()
        else:
            analysis_indices = candidate_indices
        analysis_dataset = Subset(full_train_dataset, analysis_indices)
        print(f"Analyzing a random subset of {len(analysis_indices)} candidates...")

        low_conf_idx = get_low_confidence_indices(model, analysis_dataset, device, entropy_threshold)
        high_loss_idx = get_high_loss_indices(model, analysis_dataset, device, criterion, high_loss_k_percent)
        unique_idx = get_unique_indices(model, pool_dataset, analysis_dataset, device, uniqueness_n_clusters, uniqueness_similarity_threshold)

        new_indices = set(low_conf_idx) | set(high_loss_idx) | set(unique_idx)
        print(f"Selected {len(new_indices)} new data points ({len(low_conf_idx)} low_conf, {len(high_loss_idx)} high_loss, {len(unique_idx)} unique).")

        if not new_indices:
            print("No new data selected. Ending training early.")
            break

        pool_indices.extend(list(new_indices))
        candidate_indices = [idx for idx in candidate_indices if idx not in new_indices]

        accuracy = evaluate_model(model, test_loader, device)
        print(f"Iteration {iteration + 1} | Pool Size: {len(pool_indices)} | Test Accuracy: {accuracy:.2f}%")

    print("\n--- Acural Recursion PoC Complete ---")
    final_accuracy = evaluate_model(model, test_loader, device)
    print(f'\n==================================================')
    print(f'Final Test Accuracy: {final_accuracy:.2f}%')
    print(f'Using {len(pool_indices)} / {len(full_train_dataset)} training samples.')
    print(f'Baseline Accuracy was: 98.97%')
    print(f'==================================================')

if __name__ == '__main__':
    run_acural_recursion_poc()
