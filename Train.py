
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import numpy as np

from config import (
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    LOG_FREQUENCY, MODEL_CHECKPOINT_PATTERN
)


def train_one_epoch(
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        epoch: int
) -> float:
    """
    Train model for one epoch.

    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number

    Returns:
        Average training loss (MAE) for the epoch
    """
    model.train()
    running_loss = 0.0
    num_samples = 0

    for batch_idx, (images, spacings, labels) in enumerate(train_loader):
        # Move to device
        images = images.to(device)
        spacings = spacings.to(device)
        labels = labels.to(device).unsqueeze(1)  # Shape: (batch, 1)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images, spacings)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate loss
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        num_samples += batch_size

    epoch_loss = running_loss / num_samples
    return epoch_loss


def evaluate(
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate model on validation or test set.

    Args:
        model: The neural network model
        data_loader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        Tuple of (average_loss, predictions, ground_truth)
    """
    model.eval()
    running_loss = 0.0
    num_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, spacings, labels in data_loader:
            # Move to device
            images = images.to(device)
            spacings = spacings.to(device)
            labels = labels.to(device).unsqueeze(1)

            # Forward pass
            outputs = model(images, spacings)
            loss = criterion(outputs, labels)

            # Accumulate loss and predictions
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            num_samples += batch_size

            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / num_samples
    predictions = np.concatenate(all_preds, axis=0)
    ground_truth = np.concatenate(all_labels, axis=0)

    return avg_loss, predictions, ground_truth


def train_fold(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        fold_idx: int,
        num_epochs: int = NUM_EPOCHS
) -> Dict:
    """
    Train model for one fold of cross-validation.

    Args:
        model: The neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        device: Device to train on
        fold_idx: Current fold index (0-based)
        num_epochs: Number of epochs to train

    Returns:
        Dictionary with training history and final test results
    """
    # Setup optimizer and loss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    criterion = nn.L1Loss()  # MAE Loss

    # Track best model
    best_val_mae = float('inf')
    best_model_path = MODEL_CHECKPOINT_PATTERN.format(fold=fold_idx + 1)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_val_mae': None,
        'test_mae': None
    }

    print(f"\n{'=' * 60}")
    print(f"Training Fold {fold_idx + 1}")
    print(f"{'=' * 60}")

    # Training loop
    for epoch in range(num_epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        history['train_loss'].append(train_loss)

        # Validate
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)

        # Log progress
        if (epoch + 1) % LOG_FREQUENCY == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{num_epochs} | "
                  f"Train MAE: {train_loss:6.2f} cm | "
                  f"Val MAE: {val_loss:6.2f} cm")

        # Save best model
        if val_loss < best_val_mae:
            best_val_mae = val_loss
            torch.save(model.state_dict(), best_model_path)
            if (epoch + 1) % LOG_FREQUENCY == 0:
                print(f"  → New best model saved (Val MAE: {best_val_mae:.2f} cm)")

    history['best_val_mae'] = best_val_mae
    print(f"\n  Best Validation MAE: {best_val_mae:.2f} cm")
    print(f"  Model saved to: {best_model_path}")

    # Load best model and evaluate on test set
    print(f"\n  Evaluating on test set...")
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

    history['test_mae'] = test_loss
    history['test_predictions'] = test_preds
    history['test_labels'] = test_labels

    print(f"  Final Test MAE: {test_loss:.2f} cm")
    print(f"{'=' * 60}\n")

    return history


def compute_metrics(predictions: np.ndarray, ground_truth: np.ndarray) -> Dict:
    """
    Compute evaluation metrics.

    Args:
        predictions: Model predictions
        ground_truth: True labels

    Returns:
        Dictionary with various metrics
    """
    errors = np.abs(predictions - ground_truth).flatten()

    metrics = {
        'mae': np.mean(errors),
        'std': np.std(errors),
        'median_ae': np.median(errors),
        'max_error': np.max(errors),
        'min_error': np.min(errors),
        'rmse': np.sqrt(np.mean((predictions - ground_truth) ** 2))
    }

    return metrics


def print_metrics(metrics: Dict, title: str = "Metrics"):
    """Print formatted metrics"""
    print(f"\n{title}:")
    print(f"  MAE:        {metrics['mae']:.2f} ± {metrics['std']:.2f} cm")
    print(f"  Median AE:  {metrics['median_ae']:.2f} cm")
    print(f"  RMSE:       {metrics['rmse']:.2f} cm")
    print(f"  Min Error:  {metrics['min_error']:.2f} cm")
    print(f"  Max Error:  {metrics['max_error']:.2f} cm")