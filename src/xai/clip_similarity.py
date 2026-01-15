"""
CLIP Text-Image Similarity for Emotion State Classification
"""
import numpy as np
import torch
import open_clip
from src.models.clip import get_clip


# Emotional state prompts
_PROMPTS = [
    "a relaxed emotional state",
    "a happy emotional state",
    "a stressed emotional state",
    "an anxious emotional state"
]

# Global cache for text embeddings
_TEXT_EMBEDDINGS_CACHE = None


def get_prompt_text_embeddings():
    """
    Encode text prompts using CLIP model and tokenizer.
    L2-normalize and cache the embeddings.

    Returns:
        text_embeds: Normalized text embeddings (numpy array)
    """
    global _TEXT_EMBEDDINGS_CACHE

    if _TEXT_EMBEDDINGS_CACHE is None:
        # Get CLIP model
        model, _ = get_clip()

        # Tokenize prompts
        device = next(model.parameters()).device
        text_tokens = open_clip.tokenize(_PROMPTS).to(device)

        # Encode text
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            # L2 normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Convert to numpy and cache
        _TEXT_EMBEDDINGS_CACHE = text_features.cpu().numpy()

    return _TEXT_EMBEDDINGS_CACHE


def similarity_scores(img_embed, text_embeds):
    """
    Compute similarity scores between image embedding and text embeddings.

    Args:
        img_embed: Image embedding (1D numpy array)
        text_embeds: Text embeddings (2D numpy array, shape: [num_prompts, embed_dim])

    Returns:
        list of tuples: [(label, score), ...] sorted by score descending
    """
    # Compute cosine similarity: img @ text.T
    similarities = img_embed @ text_embeds.T

    # Apply softmax to get probability distribution
    exp_sim = np.exp(similarities - np.max(similarities))  # Numerical stability
    softmax_scores = exp_sim / np.sum(exp_sim)

    # Create list of (label, score) tuples
    results = [(label, float(score)) for label, score in zip(_PROMPTS, softmax_scores)]

    # Sort by score descending
    results.sort(key=lambda x: x[1], reverse=True)

    return results
