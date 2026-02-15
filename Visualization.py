"""
Visualization utilities for model interpretation and dataset exploration
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
import random


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for visualizing model attention.
    """

    def __init__(self, model, target_layer):
        """
        Args:
            model: The neural network model
            target_layer: Layer to compute gradients for (e.g., model.backbone.layer4)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Hook to save forward pass activations"""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients"""
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, input_spacing):
        """
        Generate Class Activation Map.

        Args:
            input_image: Input tensor (1, C, H, W)
            input_spacing: Spacing tensor (1, 2)

        Returns:
            CAM heatmap as numpy array
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_image, input_spacing)

        # Backward pass
        self.model.zero_grad()
        output.backward()

        # Get weights from gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weight activations
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        # Generate heatmap
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap = heatmap / torch.max(heatmap)

        return heatmap.cpu().numpy()

    def visualize(
            self,
            input_image,
            input_spacing,
            original_image=None,
            alpha=0.4,
            colormap=cv2.COLORMAP_JET
    ):
        """
        Generate and visualize GradCAM overlay.

        Args:
            input_image: Input tensor (1, C, H, W)
            input_spacing: Spacing tensor (1, 2)
            original_image: Original image for overlay (optional)
            alpha: Transparency of heatmap overlay
            colormap: OpenCV colormap for heatmap

        Returns:
            Tuple of (heatmap, overlay_image)
        """
        # Generate CAM
        cam = self.generate_cam(input_image, input_spacing)

        # Resize to match input size
        cam = cv2.resize(cam, (input_image.shape[3], input_image.shape[2]))

        # Normalize to 0-255
        cam = (cam * 255).astype(np.uint8)

        # Apply colormap
        heatmap = cv2.applyColorMap(cam, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Create overlay
        if original_image is None:
            # Use first channel of input
            original_image = input_image[0, 0].cpu().numpy()
            original_image = (original_image * 255).astype(np.uint8)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

        overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)

        return heatmap, overlay


def visualize_dataset_samples(
        dataset,
        num_samples: int = 5,
        title_prefix: str = "Sample",
        figsize: Tuple[int, int] = (15, 3)
):
    """
    Visualize random samples from the dataset.

    Args:
        dataset: LocalizerDataset instance
        num_samples: Number of samples to visualize
        title_prefix: Prefix for subplot titles
        figsize: Figure size
    """
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    fig, axes = plt.subplots(1, num_samples, figsize=figsize)
    if num_samples == 1:
        axes = [axes]

    for idx, ax in zip(indices, axes):
        img_tensor, spacing, height = dataset[idx]

        # Convert to numpy and denormalize
        img = img_tensor[0].numpy()  # Take first channel

        ax.imshow(img, cmap='gray')
        ax.set_title(
            f"{title_prefix} {idx}\n"
            f"Height: {height:.1f} cm\n"
            f"Spacing: ({spacing[0]:.2f}, {spacing[1]:.2f})"
        )
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def visualize_predictions(
        images: np.ndarray,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        num_samples: int = 5,
        indices: Optional[List[int]] = None,
        figsize: Tuple[int, int] = (15, 3)
):
    """
    Visualize model predictions vs ground truth.

    Args:
        images: Array of images
        predictions: Model predictions
        ground_truth: True labels
        num_samples: Number of samples to show
        indices: Specific indices to visualize (optional)
        figsize: Figure size
    """
    if indices is None:
        indices = random.sample(range(len(images)), min(num_samples, len(images)))

    fig, axes = plt.subplots(1, len(indices), figsize=figsize)
    if len(indices) == 1:
        axes = [axes]

    for idx, ax in zip(indices, axes):
        img = images[idx]
        pred = predictions[idx].item() if isinstance(predictions[idx], np.ndarray) else predictions[idx]
        true = ground_truth[idx].item() if isinstance(ground_truth[idx], np.ndarray) else ground_truth[idx]
        error = abs(pred - true)

        # Handle multichannel images
        if img.ndim == 3 and img.shape[0] == 3:
            img = img[0]  # Take first channel

        ax.imshow(img, cmap='gray')
        color = 'green' if error < 5 else 'orange' if error < 10 else 'red'
        ax.set_title(
            f"True: {true:.1f} cm\n"
            f"Pred: {pred:.1f} cm\n"
            f"Error: {error:.1f} cm",
            color=color
        )
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_training_history(history: dict, fold_idx: int = None):
    """
    Plot training and validation loss curves.

    Args:
        history: Dictionary with 'train_loss' and 'val_loss' lists
        fold_idx: Fold number for title (optional)
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    epochs = range(1, len(history['train_loss']) + 1)

    ax.plot(epochs, history['train_loss'], label='Train MAE', marker='o', markersize=3)
    ax.plot(epochs, history['val_loss'], label='Val MAE', marker='s', markersize=3)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE (cm)')
    ax.set_title(f"Training History{f' - Fold {fold_idx}' if fold_idx else ''}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_error_distribution(errors: np.ndarray, bins: int = 50):
    """
    Plot histogram of prediction errors.

    Args:
        errors: Array of absolute errors
        bins: Number of histogram bins
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.hist(errors, bins=bins, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(errors), color='red', linestyle='--',
               label=f'Mean: {np.mean(errors):.2f} cm')
    ax.axvline(np.median(errors), color='green', linestyle='--',
               label=f'Median: {np.median(errors):.2f} cm')

    ax.set_xlabel('Absolute Error (cm)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Prediction Errors')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_scatter_predictions(
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        title: str = "Predictions vs Ground Truth"
):
    """
    Scatter plot of predictions vs ground truth.

    Args:
        predictions: Model predictions
        ground_truth: True labels
        title: Plot title
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.scatter(ground_truth, predictions, alpha=0.5, s=20)

    # Perfect prediction line
    min_val = min(ground_truth.min(), predictions.min())
    max_val = max(ground_truth.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

    ax.set_xlabel('True Height (cm)')
    ax.set_ylabel('Predicted Height (cm)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()