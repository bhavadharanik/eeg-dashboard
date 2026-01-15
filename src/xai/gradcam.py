"""
GradCAM (Gradient-weighted Class Activation Mapping) for CNN Interpretability

GradCAM produces visual explanations by:
1. Computing gradients of target class score w.r.t. feature maps
2. Global average pooling the gradients to get importance weights
3. Weighted combination of feature maps
4. ReLU to keep only positive contributions

This highlights which spatial regions the CNN "looks at" when making decisions.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", ICCV 2017

For EEG emotion recognition, GradCAM can reveal:
- Which frequency bands (vertical regions in spectrogram) are important
- Which time periods (horizontal regions) influence the classification
- Whether the model focuses on known EEG emotional markers
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Union
import warnings


class GradCAM:
    """
    GradCAM implementation for PyTorch CNN models.

    Usage:
        >>> gradcam = GradCAM(model, target_layer='conv2')
        >>> heatmap = gradcam.generate(image)
        >>> gradcam.remove_hooks()  # Clean up when done
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[str] = None
    ):
        """
        Initialize GradCAM.

        Args:
            model: PyTorch CNN model
            target_layer: Name of layer to visualize (default: last conv layer)
                         For EEG_CNN, this would be 'conv2'
        """
        self.model = model
        self.model.eval()

        # Storage for activations and gradients
        self.activations = None
        self.gradients = None

        # Find target layer
        if target_layer is None:
            target_layer = self._find_last_conv_layer()

        self.target_layer = target_layer

        # Register hooks
        self._register_hooks()

    def _find_last_conv_layer(self) -> str:
        """Find the name of the last convolutional layer."""
        last_conv = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = name
        if last_conv is None:
            raise ValueError("No Conv2d layer found in model")
        return last_conv

    def _get_layer(self, name: str) -> nn.Module:
        """Get a layer by name."""
        modules = dict(self.model.named_modules())
        if name not in modules:
            raise ValueError(f"Layer '{name}' not found. Available: {list(modules.keys())}")
        return modules[name]

    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        target = self._get_layer(self.target_layer)

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.forward_handle = target.register_forward_hook(forward_hook)
        self.backward_handle = target.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        """Remove registered hooks."""
        if hasattr(self, 'forward_handle'):
            self.forward_handle.remove()
        if hasattr(self, 'backward_handle'):
            self.backward_handle.remove()

    def generate(
        self,
        image: np.ndarray,
        target_class: Optional[int] = None,
        relu: bool = True
    ) -> np.ndarray:
        """
        Generate GradCAM heatmap for an input image.

        Args:
            image: Input image (224, 224, 3) RGB or (224, 224) grayscale
            target_class: Class to visualize (None = predicted class)
            relu: Apply ReLU to show only positive contributions

        Returns:
            heatmap: 2D array (224, 224) normalized to [0, 1]
        """
        # Prepare input tensor - keep RGB channels for 3-channel CNN
        if image.ndim == 2:
            # Grayscale - expand to 3 channels
            image = np.stack([image, image, image], axis=-1)

        if image.max() > 1:
            image = image.astype(np.float32) / 255.0

        # Convert to tensor: (H, W, C) -> (C, H, W) -> (1, C, H, W)
        input_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
        input_tensor.requires_grad = True

        # Forward pass
        output = self.model(input_tensor)

        # Determine target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        # Get importance weights (global average pooling of gradients)
        # gradients shape: (1, C, H, W)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of feature maps
        # activations shape: (1, C, H, W)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H, W)

        # Apply ReLU (only positive contributions)
        if relu:
            cam = F.relu(cam)

        # Convert to numpy and resize to input size
        cam = cam.squeeze().detach().numpy()  # (H, W)

        # Resize to input size (224, 224)
        cam = self._resize_cam(cam, (224, 224))

        # Normalize to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam.astype(np.float32)

    def _resize_cam(self, cam: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize CAM to target size using bilinear interpolation."""
        from PIL import Image

        # Convert to PIL Image
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam_norm = ((cam - cam_min) / (cam_max - cam_min) * 255).astype(np.uint8)
        else:
            cam_norm = np.zeros_like(cam, dtype=np.uint8)

        cam_img = Image.fromarray(cam_norm)
        cam_resized = cam_img.resize(size, Image.Resampling.BILINEAR)

        # Convert back to float and restore range
        result = np.array(cam_resized).astype(np.float32) / 255.0
        if cam_max > cam_min:
            result = result * (cam_max - cam_min) + cam_min

        return result


class GradCAMPlusPlus(GradCAM):
    """
    GradCAM++ - Improved version with better localization.

    GradCAM++ uses a weighted combination of positive partial derivatives
    as importance weights, providing better localization especially for
    multiple instances of the same class.

    Reference: Chattopadhay et al., "Grad-CAM++: Generalized Gradient-based
    Visual Explanations for Deep Convolutional Networks", WACV 2018
    """

    def generate(
        self,
        image: np.ndarray,
        target_class: Optional[int] = None,
        relu: bool = True
    ) -> np.ndarray:
        """Generate GradCAM++ heatmap."""
        # Prepare input - keep RGB channels for 3-channel CNN
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)

        if image.max() > 1:
            image = image.astype(np.float32) / 255.0

        # Convert to tensor: (H, W, C) -> (C, H, W) -> (1, C, H, W)
        input_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
        input_tensor.requires_grad = True

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        # GradCAM++ weights computation
        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)

        # Compute alpha (importance weights for each gradient)
        grad_2 = gradients ** 2
        grad_3 = gradients ** 3

        sum_activations = torch.sum(activations, dim=(2, 3), keepdim=True)
        alpha_num = grad_2
        alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-8
        alpha = alpha_num / alpha_denom

        # ReLU on gradients
        weights = torch.sum(alpha * F.relu(gradients), dim=(2, 3), keepdim=True)

        # Weighted combination
        cam = torch.sum(weights * activations, dim=1, keepdim=True)

        if relu:
            cam = F.relu(cam)

        # Convert and resize
        cam = cam.squeeze().detach().numpy()
        cam = self._resize_cam(cam, (224, 224))

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam.astype(np.float32)


def explain_cnn_gradcam(
    image: np.ndarray,
    model: nn.Module,
    target_class: Optional[int] = None,
    target_layer: Optional[str] = None,
    method: str = 'gradcam'
) -> np.ndarray:
    """
    Generate GradCAM explanation for CNN prediction.

    This is the main function to use for GradCAM-based explanations.

    Args:
        image: Input image (224, 224, 3) or (224, 224)
        model: PyTorch CNN model
        target_class: Class to explain (None = predicted class)
        target_layer: Layer to visualize (None = last conv layer)
        method: 'gradcam' or 'gradcam++'

    Returns:
        heatmap: 2D array (224, 224) with values in [0, 1]

    Example:
        >>> from src.models.cnn import load_cnn_model
        >>> model = load_cnn_model("models/cnn_baseline.pt")
        >>> heatmap = explain_cnn_gradcam(image, model)
        >>> # Overlay on original image for visualization
    """
    if method == 'gradcam':
        explainer = GradCAM(model, target_layer)
    elif method == 'gradcam++':
        explainer = GradCAMPlusPlus(model, target_layer)
    else:
        warnings.warn(f"Unknown method '{method}'. Using gradcam.")
        explainer = GradCAM(model, target_layer)

    try:
        heatmap = explainer.generate(image, target_class)
    finally:
        explainer.remove_hooks()

    return heatmap


def generate_layerwise_gradcam(
    image: np.ndarray,
    model: nn.Module,
    target_class: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Generate GradCAM heatmaps for all convolutional layers.

    Useful for understanding how different layers contribute to the decision.
    Earlier layers capture low-level features (edges, textures), while
    later layers capture high-level patterns.

    Args:
        image: Input image
        model: PyTorch model
        target_class: Class to explain

    Returns:
        Dictionary mapping layer names to heatmaps
    """
    heatmaps = {}

    # Find all conv layers
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append(name)

    # Generate heatmap for each layer
    for layer_name in conv_layers:
        try:
            heatmap = explain_cnn_gradcam(
                image, model, target_class, target_layer=layer_name
            )
            heatmaps[layer_name] = heatmap
        except Exception as e:
            warnings.warn(f"Failed to generate GradCAM for {layer_name}: {e}")

    return heatmaps


def combine_explanations(
    image: np.ndarray,
    model: nn.Module,
    target_class: Optional[int] = None,
    methods: List[str] = ['gradcam', 'gradcam++']
) -> np.ndarray:
    """
    Combine multiple explanation methods for more robust attribution.

    Args:
        image: Input image
        model: PyTorch model
        target_class: Class to explain
        methods: List of methods to combine

    Returns:
        Combined heatmap (average of all methods)
    """
    heatmaps = []

    for method in methods:
        try:
            heatmap = explain_cnn_gradcam(image, model, target_class, method=method)
            heatmaps.append(heatmap)
        except Exception as e:
            warnings.warn(f"Method {method} failed: {e}")

    if not heatmaps:
        return np.zeros((224, 224), dtype=np.float32)

    combined = np.mean(heatmaps, axis=0)

    # Renormalize
    if combined.max() > combined.min():
        combined = (combined - combined.min()) / (combined.max() - combined.min())

    return combined.astype(np.float32)


def analyze_eeg_regions(
    heatmap: np.ndarray,
    spectrogram_info: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Analyze which regions of the spectrogram are most important.

    For EEG spectrograms, this can reveal:
    - Which frequency bands (delta, theta, alpha, beta, gamma) are important
    - Which time periods show significant activity

    Args:
        heatmap: GradCAM heatmap (224, 224)
        spectrogram_info: Optional dict with 'freq_range' and 'time_range'

    Returns:
        Dictionary with regional importance scores
    """
    h, w = heatmap.shape

    # Divide into frequency bands (vertical regions)
    # Assuming standard spectrogram with low freq at bottom
    n_freq_bands = 5  # delta, theta, alpha, beta, gamma
    freq_band_size = h // n_freq_bands

    band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    freq_importance = {}

    for i, band in enumerate(band_names):
        # Higher index = higher frequency (top of image)
        start_row = h - (i + 1) * freq_band_size
        end_row = h - i * freq_band_size
        band_region = heatmap[max(0, start_row):end_row, :]
        freq_importance[band] = float(np.mean(band_region))

    # Divide into time periods (horizontal regions)
    n_time_periods = 4  # early, mid-early, mid-late, late
    time_period_size = w // n_time_periods

    time_names = ['early', 'mid_early', 'mid_late', 'late']
    time_importance = {}

    for i, period in enumerate(time_names):
        start_col = i * time_period_size
        end_col = (i + 1) * time_period_size
        time_region = heatmap[:, start_col:end_col]
        time_importance[period] = float(np.mean(time_region))

    return {
        'frequency_bands': freq_importance,
        'time_periods': time_importance,
        'overall_mean': float(np.mean(heatmap)),
        'max_activation': float(np.max(heatmap)),
        'activation_coverage': float(np.mean(heatmap > 0.5))  # % of high activation
    }


if __name__ == "__main__":
    import sys
    sys.path.append('.')

    from src.models.cnn import load_cnn_model

    print("Loading CNN model...")
    model = load_cnn_model("models/cnn_baseline.pt")

    print("Creating test image...")
    test_image = np.random.rand(224, 224, 3).astype(np.float32)

    print("\nTesting GradCAM...")
    heatmap_gc = explain_cnn_gradcam(test_image, model, method='gradcam')
    print(f"  Shape: {heatmap_gc.shape}")
    print(f"  Range: [{heatmap_gc.min():.4f}, {heatmap_gc.max():.4f}]")

    print("\nTesting GradCAM++...")
    heatmap_gcpp = explain_cnn_gradcam(test_image, model, method='gradcam++')
    print(f"  Shape: {heatmap_gcpp.shape}")
    print(f"  Range: [{heatmap_gcpp.min():.4f}, {heatmap_gcpp.max():.4f}]")

    print("\nAnalyzing EEG regions...")
    analysis = analyze_eeg_regions(heatmap_gc)
    print("  Frequency band importance:")
    for band, score in analysis['frequency_bands'].items():
        print(f"    {band}: {score:.4f}")

    print("\nDone! GradCAM working correctly.")
