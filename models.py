import torch
import torch.nn as nn
import torchvision.models as models

# --- Custom CNNs from previous steps ---

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 64 * 7 * 7)
        return self.fc_layers(x)

    def get_features(self, x):
        return self.conv_layers(x).view(-1, 64 * 7 * 7)

class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super(CIFAR10_CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(nn.Linear(64 * 8 * 8, 512), nn.ReLU(), nn.Linear(512, 10))

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 64 * 8 * 8)
        return self.fc_layers(x)

    def get_features(self, x):
        return self.conv_layers(x).view(-1, 64 * 8 * 8)

# --- Foundational Model Integration ---

class ResNetWrapper(nn.Module):
    """A wrapper to make torchvision's ResNet compatible with our `get_features` method."""
    def __init__(self, resnet_model):
        super(ResNetWrapper, self).__init__()
        # The original model
        self.resnet = resnet_model
        # Create a feature extractor by removing the final classification layer (self.resnet.fc)
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])

    def forward(self, x):
        return self.resnet(x)

    def get_features(self, x):
        # Pass data through the feature extractor and flatten
        features = self.feature_extractor(x)
        return torch.flatten(features, 1)

def get_model(name, num_classes=10, pretrained=True):
    """
    Model factory function.
    - name: 'MNIST_CNN', 'CIFAR10_CNN', or a torchvision model name like 'resnet50'.
    - num_classes: Number of output classes.
    - pretrained: Whether to load pretrained weights (for torchvision models).
    """
    if name == 'MNIST_CNN':
        return MNIST_CNN()
    elif name == 'CIFAR10_CNN':
        return CIFAR10_CNN()

    # Logic for torchvision models
    elif name in models.__dict__:
        # Load the model from torchvision
        model_func = models.__dict__[name]
        model = model_func(weights='DEFAULT' if pretrained else None)

        # Freeze parameters if using pretrained model
        if pretrained:
            for param in model.parameters():
                param.requires_grad = False

        # Replace the classifier head for fine-tuning
        if 'resnet' in name or 'resnext' in name:
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        elif 'densenet' in name:
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)
        elif 'vgg' in name:
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
        else:
            raise NotImplementedError(f"Fine-tuning not implemented for {name}. Please add it to `models.py`.")

        # Wrap the model to ensure it has our custom `get_features` method
        return ResNetWrapper(model)
    else:
        raise ValueError(f"Model {name} not recognized.")
