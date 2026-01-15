"""
Model definitions for EEG emotion recognition.

Available models:
- CNN: Simple 2-layer CNN for spectrogram classification
- CLIP: Vision-language model wrapper for transfer learning
"""

from src.models.cnn import EEG_CNN, load_cnn_model, predict_cnn
from src.models.clip import get_clip, embed_image

__all__ = [
    "EEG_CNN",
    "load_cnn_model",
    "predict_cnn",
    "get_clip",
    "embed_image",
]
