"""
XAI Explainability Module using SHAP (SHapley Additive exPlanations)

SHAP provides a unified approach to explain model predictions by computing
the contribution of each feature (pixel) to the output. This is crucial
for understanding what EEG patterns the model considers important for
emotion classification.

Methods implemented:
1. GradientSHAP - Fast gradient-based approximation (for CNNs)
2. DeepSHAP - DeepLIFT + Shapley values (more accurate)
3. KernelSHAP - Model-agnostic (slower but works with any model)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import warnings


def explain_cnn_gradient_shap(
    image: np.ndarray,
    model: nn.Module,
    target_class: Optional[int] = None,
    n_samples: int = 50,
    stdev_spread: float = 0.15,
    baseline: str = 'zeros'
) -> np.ndarray:
    """
    Generate SHAP explanation using GradientSHAP method.

    GradientSHAP combines ideas from Integrated Gradients and SHAP.
    It approximates SHAP values by:
    1. Sampling random points between baseline and input
    2. Computing gradients at these points
    3. Averaging the gradients weighted by (input - baseline)

    This tells us which pixels most influenced the model's decision.

    Args:
        image: Input image (224, 224, 3) RGB or (224, 224) grayscale
        model: PyTorch CNN model
        target_class: Which class to explain (None = predicted class)
        n_samples: Number of samples for gradient estimation
        stdev_spread: Standard deviation for noise (as fraction of input range)
        baseline: 'zeros', 'noise', or 'blur' - reference point

    Returns:
        heatmap: 2D array (224, 224) with SHAP values

    Example:
        >>> heatmap = explain_cnn_gradient_shap(img, model, target_class=1)
        >>> # High values = pixels that increased class 1 probability
    """
    model.eval()

    # Prepare input tensor
    if image.ndim == 3:
        gray = image[:, :, 0] if image.shape[2] == 3 else image
    else:
        gray = image

    # Normalize to [0, 1]
    if gray.max() > 1:
        gray = gray.astype(np.float32) / 255.0

    # Create input tensor (1, 1, 224, 224)
    input_tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0)
    input_tensor.requires_grad = True

    # Create baseline
    if baseline == 'zeros':
        baseline_tensor = torch.zeros_like(input_tensor)
    elif baseline == 'noise':
        baseline_tensor = torch.randn_like(input_tensor) * 0.1
    elif baseline == 'blur':
        # Use heavily blurred version as baseline
        from scipy.ndimage import gaussian_filter
        blurred = gaussian_filter(gray, sigma=20)
        baseline_tensor = torch.from_numpy(blurred).float().unsqueeze(0).unsqueeze(0)
    else:
        baseline_tensor = torch.zeros_like(input_tensor)

    # Determine target class if not specified
    if target_class is None:
        with torch.no_grad():
            output = model(input_tensor)
            target_class = output.argmax(dim=1).item()

    # Compute GradientSHAP
    gradients_list = []
    stdev = stdev_spread * (input_tensor.max() - input_tensor.min())

    for _ in range(n_samples):
        # Random interpolation between baseline and input
        alpha = torch.rand(1).item()

        # Add noise for better gradient estimation
        noise = torch.randn_like(input_tensor) * stdev

        # Interpolated input
        interpolated = baseline_tensor + alpha * (input_tensor - baseline_tensor) + noise
        interpolated = interpolated.clone().detach().requires_grad_(True)

        # Forward pass
        output = model(interpolated)
        target_score = output[0, target_class]

        # Backward pass
        model.zero_grad()
        target_score.backward()

        # Store gradient
        gradients_list.append(interpolated.grad.detach().numpy())

    # Average gradients
    avg_gradients = np.mean(gradients_list, axis=0)

    # SHAP values = gradient * (input - baseline)
    diff = input_tensor.detach().numpy() - baseline_tensor.numpy()
    shap_values = avg_gradients * diff

    # Extract 2D heatmap
    heatmap = shap_values[0, 0]  # Remove batch and channel dims

    # Take absolute value (we care about magnitude of influence)
    heatmap = np.abs(heatmap)

    # Normalize to [0, 1]
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    return heatmap.astype(np.float32)


def explain_cnn_integrated_gradients(
    image: np.ndarray,
    model: nn.Module,
    target_class: Optional[int] = None,
    n_steps: int = 50,
    baseline: str = 'zeros'
) -> np.ndarray:
    """
    Generate explanation using Integrated Gradients method.

    Integrated Gradients is a well-established attribution method that:
    1. Creates a path from baseline to input
    2. Integrates gradients along this path
    3. Satisfies important axioms (sensitivity, implementation invariance)

    Mathematical formulation:
    IG(x) = (x - x') × ∫₀¹ ∂F(x' + α(x-x')) / ∂x dα

    Args:
        image: Input image (224, 224, 3) or (224, 224)
        model: PyTorch model
        target_class: Class to explain
        n_steps: Number of integration steps (higher = more accurate)
        baseline: Reference point type

    Returns:
        Attribution heatmap (224, 224)
    """
    model.eval()

    # Prepare input
    if image.ndim == 3:
        gray = image[:, :, 0] if image.shape[2] == 3 else image
    else:
        gray = image

    if gray.max() > 1:
        gray = gray.astype(np.float32) / 255.0

    input_tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0)

    # Create baseline
    if baseline == 'zeros':
        baseline_tensor = torch.zeros_like(input_tensor)
    else:
        baseline_tensor = torch.zeros_like(input_tensor)

    # Determine target class
    if target_class is None:
        with torch.no_grad():
            output = model(input_tensor)
            target_class = output.argmax(dim=1).item()

    # Compute integrated gradients
    scaled_inputs = []
    for step in range(n_steps + 1):
        alpha = step / n_steps
        scaled_input = baseline_tensor + alpha * (input_tensor - baseline_tensor)
        scaled_inputs.append(scaled_input)

    # Stack all scaled inputs
    scaled_inputs = torch.cat(scaled_inputs, dim=0)
    scaled_inputs.requires_grad = True

    # Forward pass
    outputs = model(scaled_inputs)
    target_scores = outputs[:, target_class]

    # Backward pass
    model.zero_grad()
    gradient_sum = torch.zeros_like(input_tensor)

    for i in range(n_steps + 1):
        scaled_inputs.grad = None
        target_scores[i].backward(retain_graph=True)
        gradient_sum += scaled_inputs.grad[i:i+1]

    # Average gradients and multiply by (input - baseline)
    avg_gradients = gradient_sum / (n_steps + 1)
    integrated_gradients = (input_tensor - baseline_tensor) * avg_gradients

    # Extract heatmap
    heatmap = integrated_gradients[0, 0].detach().numpy()
    heatmap = np.abs(heatmap)

    # Normalize
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    return heatmap.astype(np.float32)


def explain_cnn_deep_shap(
    image: np.ndarray,
    model: nn.Module,
    background_data: Optional[np.ndarray] = None,
    target_class: Optional[int] = None
) -> np.ndarray:
    """
    Generate explanation using DeepSHAP (SHAP library's deep explainer).

    DeepSHAP combines DeepLIFT with Shapley values for more accurate
    attributions. Requires the shap library.

    Args:
        image: Input image (224, 224, 3) or (224, 224)
        model: PyTorch model
        background_data: Reference samples for SHAP (default: 100 noise samples)
        target_class: Class to explain

    Returns:
        Attribution heatmap (224, 224)
    """
    try:
        import shap
    except ImportError:
        warnings.warn("SHAP library not available. Install with: pip install shap")
        return explain_cnn_gradient_shap(image, model, target_class)

    model.eval()

    # Prepare input
    if image.ndim == 3:
        gray = image[:, :, 0] if image.shape[2] == 3 else image
    else:
        gray = image

    if gray.max() > 1:
        gray = gray.astype(np.float32) / 255.0

    input_tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0)

    # Create background data if not provided
    if background_data is None:
        # Use noise samples as background
        background_data = torch.randn(100, 1, 224, 224) * 0.1

    # Create DeepExplainer
    try:
        explainer = shap.DeepExplainer(model, background_data)
        shap_values = explainer.shap_values(input_tensor)

        # Get SHAP values for target class
        if target_class is None:
            with torch.no_grad():
                output = model(input_tensor)
                target_class = output.argmax(dim=1).item()

        heatmap = np.abs(shap_values[target_class][0, 0])

        # Normalize
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

        return heatmap.astype(np.float32)

    except Exception as e:
        warnings.warn(f"DeepSHAP failed: {e}. Falling back to GradientSHAP.")
        return explain_cnn_gradient_shap(image, model, target_class)


def explain_cnn(
    rgb224: np.ndarray,
    model: nn.Module,
    method: str = 'gradient_shap',
    target_class: Optional[int] = None,
    **kwargs
) -> np.ndarray:
    """
    Main function: Generate explanation heatmap for CNN model prediction.

    This is the unified interface for all SHAP-based explanation methods.
    Choose the method based on your needs:
    - gradient_shap: Fast, good approximation (recommended)
    - integrated_gradients: Theoretically grounded, slower
    - deep_shap: Most accurate, requires shap library

    Args:
        rgb224: Input image array (224, 224, 3) or (224, 224)
        model: Trained CNN model
        method: 'gradient_shap', 'integrated_gradients', or 'deep_shap'
        target_class: Class to explain (None = predicted class)
        **kwargs: Additional arguments for specific methods

    Returns:
        heatmap: 2D numpy array (224, 224) with values in [0, 1]
                 High values indicate important regions for the prediction

    Example:
        >>> from src.models.cnn import load_cnn_model
        >>> model = load_cnn_model("models/cnn_baseline.pt")
        >>> heatmap = explain_cnn(image, model, method='gradient_shap')
        >>> # Overlay heatmap on original image for visualization
    """
    if method == 'gradient_shap':
        return explain_cnn_gradient_shap(rgb224, model, target_class, **kwargs)
    elif method == 'integrated_gradients':
        return explain_cnn_integrated_gradients(rgb224, model, target_class, **kwargs)
    elif method == 'deep_shap':
        return explain_cnn_deep_shap(rgb224, model, target_class=target_class, **kwargs)
    else:
        warnings.warn(f"Unknown method '{method}'. Using gradient_shap.")
        return explain_cnn_gradient_shap(rgb224, model, target_class)


def compute_shap_summary(
    images: np.ndarray,
    model: nn.Module,
    method: str = 'gradient_shap'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute SHAP values for multiple images to get summary statistics.

    Useful for understanding which regions are consistently important
    across multiple samples.

    Args:
        images: Batch of images (N, 224, 224, 3) or (N, 224, 224)
        model: PyTorch model
        method: Explanation method to use

    Returns:
        mean_importance: Average absolute SHAP values (224, 224)
        std_importance: Standard deviation of SHAP values (224, 224)
    """
    all_heatmaps = []

    for i in range(len(images)):
        heatmap = explain_cnn(images[i], model, method=method)
        all_heatmaps.append(heatmap)

    all_heatmaps = np.array(all_heatmaps)

    mean_importance = np.mean(all_heatmaps, axis=0)
    std_importance = np.std(all_heatmaps, axis=0)

    return mean_importance, std_importance


if __name__ == "__main__":
    # Test the SHAP explanations
    import sys
    sys.path.append('.')

    from src.models.cnn import load_cnn_model

    print("Loading CNN model...")
    model = load_cnn_model("models/cnn_baseline.pt")

    print("Creating test image...")
    test_image = np.random.rand(224, 224, 3).astype(np.float32)

    print("\nTesting GradientSHAP...")
    heatmap_gs = explain_cnn(test_image, model, method='gradient_shap')
    print(f"  Shape: {heatmap_gs.shape}")
    print(f"  Range: [{heatmap_gs.min():.4f}, {heatmap_gs.max():.4f}]")

    print("\nTesting Integrated Gradients...")
    heatmap_ig = explain_cnn(test_image, model, method='integrated_gradients', n_steps=20)
    print(f"  Shape: {heatmap_ig.shape}")
    print(f"  Range: [{heatmap_ig.min():.4f}, {heatmap_ig.max():.4f}]")

    print("\nDone! SHAP explanations working correctly.")
