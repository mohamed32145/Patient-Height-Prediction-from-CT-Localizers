import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Callable, Dict, Optional, Tuple
import numpy as np

from config import (
    COSINE_ETA_MIN,
    COSINE_T_MAX,
    LEARNING_RATE,
    LOG_FREQUENCY,
    LOSS_NAME,
    MODEL_CHECKPOINT_PATTERN,
    NUM_EPOCHS,
    USE_COSINE_ANNEALING,
    WEIGHT_DECAY,
)


def train_one_epoch(
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    num_samples = 0

    for images, spacings, labels in train_loader:
        images = images.to(device)
        spacings = spacings.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images, spacings)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        num_samples += batch_size

    return running_loss / max(num_samples, 1)


def evaluate(
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    num_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, spacings, labels in data_loader:
            images = images.to(device)
            spacings = spacings.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images, spacings)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            num_samples += batch_size

            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / max(num_samples, 1)
    predictions = np.concatenate(all_preds, axis=0)
    ground_truth = np.concatenate(all_labels, axis=0)
    mae = np.mean(np.abs(predictions - ground_truth))

    return avg_loss, mae, predictions, ground_truth


def train_fold(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
               device: torch.device, fold_idx: int, num_epochs: int = NUM_EPOCHS,
               learning_rate: float = LEARNING_RATE, weight_decay: float = WEIGHT_DECAY,
               cosine_eta_min: float = COSINE_ETA_MIN, use_cosine_annealing: bool = USE_COSINE_ANNEALING,
               trial_callback: Optional[Callable[[int, float], None]] = None) -> Dict:
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if LOSS_NAME.lower() == 'mse':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported LOSS_NAME: {LOSS_NAME}")

    scheduler = None
    if use_cosine_annealing:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=min(COSINE_T_MAX, num_epochs),
            eta_min=cosine_eta_min,
        )

    best_val_mse = float('inf')
    best_model_path = MODEL_CHECKPOINT_PATTERN.format(fold=fold_idx + 1)

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'best_val_loss': None,
        'test_loss': None,
        'test_mae': None,
    }

    print(f"\n{'=' * 60}")
    print(f"Training Fold {fold_idx + 1}")
    print(f"{'=' * 60}")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_loss)

        val_loss, val_mae, _, _ = evaluate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)

        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % LOG_FREQUENCY == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch + 1:3d}/{num_epochs} | "
                  f"Train MSE: {train_loss:8.3f} | "
                  f"Val MSE: {val_loss:8.3f} | "
                  f"Val MAE: {val_mae:6.2f} cm | "
                  f"LR: {current_lr:.2e}")

        if trial_callback is not None:
            trial_callback(epoch, val_mae)

        if val_loss < best_val_mse:
            best_val_mse = val_loss
            torch.save(model.state_dict(), best_model_path)

    history['best_val_loss'] = best_val_mse
    print(f"\n  Best Validation MSE: {best_val_mse:.3f}")
    print(f"  Model saved to: {best_model_path}")

    print(f"\n  Evaluating on test set...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_mae, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

    history['test_loss'] = test_loss
    history['test_mae'] = test_mae
    history['test_predictions'] = test_preds
    history['test_labels'] = test_labels

    print(f"  Final Test MSE: {test_loss:.3f}")
    print(f"  Final Test MAE: {test_mae:.2f} cm")
    print(f"{'=' * 60}\n")

    return history


def compute_metrics(predictions: np.ndarray, ground_truth: np.ndarray) -> Dict:
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
    print(f"\n{title}:")
    print(f"  MAE:        {metrics['mae']:.2f} ± {metrics['std']:.2f} cm")
    print(f"  Median AE:  {metrics['median_ae']:.2f} cm")
    print(f"  RMSE:       {metrics['rmse']:.2f} cm")
    print(f"  Min Error:  {metrics['min_error']:.2f} cm")
    print(f"  Max Error:  {metrics['max_error']:.2f} cm")
