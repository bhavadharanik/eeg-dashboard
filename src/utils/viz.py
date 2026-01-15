"""
Visualization utilities for heatmaps and similarity scores
"""
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import cm
import streamlit as st


def overlay_heatmap(rgb224, heatmap):
    """
    Overlay heatmap onto RGB image using alpha compositing.

    Args:
        rgb224: Base RGB image (224x224x3) as numpy array
        heatmap: 2D heatmap array (224x224)

    Returns:
        np.array: RGB image with heatmap overlay (224x224x3)
    """
    # Ensure heatmap is 2D
    if heatmap.ndim != 2:
        heatmap = heatmap.squeeze()

    # Check for valid heatmap values
    heatmap_min = float(heatmap.min())
    heatmap_max = float(heatmap.max())

    # Normalize heatmap to [0, 1]
    if heatmap_max - heatmap_min > 1e-8:
        heatmap_norm = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
    else:
        # If heatmap is uniform, create a gradient for visibility
        heatmap_norm = np.ones_like(heatmap) * 0.5

    # Convert base image to PIL Image (RGBA)
    if rgb224.dtype == np.float32 or rgb224.dtype == np.float64:
        if rgb224.max() <= 1.0:
            rgb224 = (rgb224 * 255).astype(np.uint8)
        else:
            rgb224 = rgb224.astype(np.uint8)

    base_image = Image.fromarray(rgb224).convert("RGBA")
    img_width, img_height = base_image.size

    # Resize heatmap if needed to match base image
    if heatmap_norm.shape[0] != img_height or heatmap_norm.shape[1] != img_width:
        heatmap_pil = Image.fromarray((heatmap_norm * 255).astype(np.uint8))
        heatmap_pil = heatmap_pil.resize((img_width, img_height), Image.Resampling.BILINEAR)
        heatmap_norm = np.array(heatmap_pil).astype(np.float32) / 255.0

    # Apply colormap to heatmap (using 'jet' colormap)
    colormap = plt.colormaps.get_cmap('jet')
    heatmap_colored = colormap(heatmap_norm)  # Returns RGBA in [0, 1]

    # Convert to uint8 and create PIL Image
    heatmap_rgba = (heatmap_colored * 255).astype(np.uint8)
    heatmap_image = Image.fromarray(heatmap_rgba, mode='RGBA')

    # Set alpha channel based on heatmap intensity (more visible for higher values)
    # Use intensity-weighted alpha: low values more transparent, high values more opaque
    alpha_values = (heatmap_norm * 180 + 40).astype(np.uint8)  # Range: 40-220
    alpha = Image.fromarray(alpha_values)
    heatmap_image.putalpha(alpha)

    # Alpha composite: overlay heatmap onto base image
    result = Image.alpha_composite(base_image, heatmap_image)

    # Convert back to RGB numpy array
    result_rgb = np.array(result.convert("RGB"))

    return result_rgb


def show_similarity_bars(label_scores):
    """
    Create Plotly bar chart showing similarity scores.

    Args:
        label_scores: List of tuples [(label, score), ...]

    Returns:
        plotly.graph_objects.Figure: Bar chart figure
    """
    # Extract labels and scores
    labels = [label for label, _ in label_scores]
    scores = [score for _, score in label_scores]

    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=scores,
            marker=dict(
                color=scores,
                colorscale='Viridis',
                showscale=False
            ),
            text=[f"{score:.3f}" for score in scores],
            textposition='auto',
        )
    ])

    # Update layout
    fig.update_layout(
        title="Prompt Similarity",
        xaxis_title="Emotional State",
        yaxis_title="Similarity Score",
        yaxis_range=[0, 1],
        template="plotly_white",
        height=400,
        showlegend=False
    )

    return fig


def visualize_eeg_channels(sample_data, channel_names):
    """
    Create a visualization of EEG channel values.

    Args:
        sample_data: 1D array of EEG channel values (32,)
        channel_names: List of channel names

    Returns:
        plotly.graph_objects.Figure: Channel visualization
    """
    # Create bar chart of channel values
    fig = go.Figure(data=[
        go.Bar(
            x=channel_names,
            y=sample_data,
            marker=dict(
                color=sample_data,
                colorscale='RdBu_r',
                showscale=True,
                colorbar=dict(title="Amplitude")
            ),
            text=[f"{val:.2f}" for val in sample_data],
            textposition='auto',
        )
    ])

    # Update layout
    fig.update_layout(
        title="EEG Channel Amplitudes",
        xaxis_title="Channel",
        yaxis_title="Amplitude",
        template="plotly_white",
        height=500,
        showlegend=False,
        xaxis=dict(tickangle=-45)
    )

    return fig


def create_eeg_topographic_map(sample_data, channel_names):
    """
    Create a simple topographic-style heatmap visualization.

    Args:
        sample_data: 1D array of EEG channel values (32,)
        channel_names: List of channel names

    Returns:
        np.ndarray: RGB image array (224, 224, 3)
    """
    # Create a simple grid representation (8x4 grid for 32 channels)
    grid = np.zeros((8, 4))
    for i, val in enumerate(sample_data):
        row = i // 4
        col = i % 4
        grid[row, col] = val

    # Normalize to [0, 1]
    grid_norm = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)

    # Apply colormap
    colormap = plt.colormaps.get_cmap('RdBu_r')
    colored_grid = colormap(grid_norm)

    # Convert to PIL and resize to 224x224
    img = Image.fromarray((colored_grid[:, :, :3] * 255).astype(np.uint8))
    img_resized = img.resize((224, 224), Image.Resampling.NEAREST)

    # Convert to RGB array
    rgb_array = np.array(img_resized)

    return rgb_array


def plot_channel_time_series(sample_data, channel_names):
    """
    Create a line plot showing the relative values across channels.

    Args:
        sample_data: 1D array of EEG channel values (32,)
        channel_names: List of channel names

    Returns:
        plotly.graph_objects.Figure: Line plot
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(len(sample_data))),
        y=sample_data,
        mode='lines+markers',
        name='Channel Values',
        line=dict(color='royalblue', width=2),
        marker=dict(size=6)
    ))

    # Update layout
    fig.update_layout(
        title="EEG Channel Values Across Channels",
        xaxis_title="Channel Index",
        yaxis_title="Amplitude",
        template="plotly_white",
        height=400,
        hovermode='x unified'
    )

    # Add channel names as hover text
    fig.update_traces(
        text=channel_names,
        hovertemplate='<b>%{text}</b><br>Amplitude: %{y:.2f}<extra></extra>'
    )

    return fig
