"""
Explainable AI (XAI) modules for model interpretation.

Available methods:
- GradCAM: Gradient-weighted Class Activation Mapping
- SHAP: SHapley Additive exPlanations
- CLIP Similarity: Text-image similarity analysis
"""

from src.xai.gradcam import explain_cnn_gradcam, analyze_eeg_regions
from src.xai.clip_similarity import get_prompt_text_embeddings, similarity_scores

__all__ = [
    "explain_cnn_gradcam",
    "analyze_eeg_regions",
    "get_prompt_text_embeddings",
    "similarity_scores",
]
