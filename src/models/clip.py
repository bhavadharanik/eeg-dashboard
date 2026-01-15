"""
CLIP Model Wrapper using open-clip-torch
"""
import numpy as np
import torch
from PIL import Image
import open_clip
import streamlit as st


@st.cache_resource(show_spinner=False)
def get_clip():
    """
    Load CLIP model and preprocess transform, cached using Streamlit's cache.

    Returns:
        tuple: (model, preprocess) - CLIP model and preprocessing transform
    """
    import warnings

    try:
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Suppress QuickGELU mismatch warning (cosmetic, doesn't affect functionality)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="QuickGELU mismatch")

            # Load model with explicit device placement
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32",
                pretrained="openai",
                device=device
            )
        model.eval()  # Set to evaluation mode

        return model, preprocess

    except Exception as e:
        st.error(f"Failed to load CLIP model: {str(e)}")
        raise


def embed_image(rgb224, model, preprocess):
    """
    Convert numpy array to PIL image, preprocess, and encode to normalized embedding.

    Args:
        rgb224: Numpy array image (224x224x3) with values in [0, 255] or [0, 1]
        model: CLIP model
        preprocess: CLIP preprocessing transform

    Returns:
        embedding: Normalized image embedding (numpy array)
    """
    # Convert numpy array to PIL Image
    if rgb224.dtype == np.float32 or rgb224.dtype == np.float64:
        # If float, assume [0, 1] range and convert to [0, 255]
        if rgb224.max() <= 1.0:
            rgb224 = (rgb224 * 255).astype(np.uint8)
        else:
            rgb224 = rgb224.astype(np.uint8)

    pil_image = Image.fromarray(rgb224)

    # Apply preprocessing
    image_tensor = preprocess(pil_image).unsqueeze(0)  # Add batch dimension

    # Move to same device as model
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    # Encode image
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        # Normalize embedding
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # Convert to numpy and remove batch dimension
    embedding = image_features.cpu().numpy().squeeze()

    return embedding
