"""
CNN model for EEG emotion classification.

Lightweight CNN baseline for comparison with CLIP-based approach.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EEG_CNN(nn.Module):
    """
    Lightweight CNN architecture for EEG spectrogram emotion classification.

    Architecture:
    - Conv Block 1: Conv2d(32) -> BatchNorm -> ReLU -> MaxPool
    - Conv Block 2: Conv2d(64) -> BatchNorm -> ReLU -> MaxPool
    - Conv Block 3: Conv2d(128) -> BatchNorm -> ReLU -> MaxPool
    - Global Average Pooling -> 128-dim feature vector
    - MLP Classifier: Dense(256) -> Dense(128) -> Dense(2)

    This provides comparable architecture to CLIP+MLP for fair comparison.
    """

    def __init__(self, num_classes=2):
        super(EEG_CNN, self).__init__()

        # Conv Block 1: 224x224 -> 112x112
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 2: 112x112 -> 56x56
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 3: 56x56 -> 28x28
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Global Average Pooling -> 128-dim feature vector
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # MLP Classifier (same structure as CLIP classifier for fair comparison)
        self.fc1 = nn.Linear(128, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Global Average Pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 128)

        # MLP Classifier
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)

        return x

    def extract_features(self, x):
        """
        Extract 128-dim feature vector (before classifier).

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Feature tensor of shape (batch_size, 128)
        """
        # Conv blocks
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Global Average Pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        return x


def load_cnn_model(path: str, num_classes: int = 2) -> EEG_CNN:
    """
    Load CNN model from checkpoint or return randomly initialized model.

    Args:
        path: Path to model checkpoint (.pt file)
        num_classes: Number of output classes (default: 2)

    Returns:
        EEG_CNN model in evaluation mode
    """
    # Initialize model
    model = EEG_CNN(num_classes=num_classes)

    # Load weights if file exists
    if os.path.exists(path):
        try:
            state_dict = torch.load(path, map_location=torch.device('cpu'), weights_only=True)
            model.load_state_dict(state_dict)
            print(f"Loaded model weights from {path}")
        except Exception as e:
            print(f"Warning: Could not load model from {path}: {e}")
            print("Using randomly initialized model instead.")
    else:
        print(f"Model file not found at {path}. Using randomly initialized model.")

    # Set to evaluation mode
    model.eval()

    return model


def predict_cnn(rgb224: np.ndarray, model: EEG_CNN) -> tuple[str, dict[str, float]]:
    """
    Predict emotion from RGB spectrogram image.

    Args:
        rgb224: RGB numpy array of shape (224, 224, 3) with values in [0, 255]
        model: Trained EEG_CNN model

    Returns:
        Tuple of (predicted_label, probabilities_dict)
        - predicted_label: String label ("Negative" or "Positive")
        - probabilities_dict: Dictionary mapping class names to probabilities
    """
    # Class labels
    class_labels = ["Negative", "Positive"]

    # Convert to tensor and normalize to [0, 1]
    # Input: (224, 224, 3) -> (3, 224, 224)
    tensor = torch.from_numpy(rgb224).float().permute(2, 0, 1) / 255.0

    # Add batch dimension: (3, 224, 224) -> (1, 3, 224, 224)
    tensor = tensor.unsqueeze(0)

    # Run inference
    model.eval()
    with torch.no_grad():
        logits = model(tensor)

        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)

        # Get predicted class
        pred_idx = torch.argmax(probs, dim=1).item()
        predicted_label = class_labels[pred_idx]

        # Create probabilities dictionary
        probs_dict = {
            class_labels[i]: probs[0, i].item()
            for i in range(len(class_labels))
        }

    return predicted_label, probs_dict
