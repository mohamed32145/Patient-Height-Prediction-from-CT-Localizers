import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple

from config import (
    ADAMW_BETAS,
    LEARNING_RATE,
    LOG_FREQUENCY,
    LOSS_NAME,
    LR_GAMMA,
    LR_STEP_SIZE,
    MODEL_CHECKPOINT_PATTERN,
    NUM_EPOCHS,
    WEIGHT_DECAY,
)


def train_one_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> float:
    model.train()
    running_loss = 0.0
    num_samples = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        bsz = images.size(0)
        running_loss += loss.item() * bsz
        num_samples += bsz

    return running_loss / max(num_samples, 1)


def evaluate(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    num_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            bsz = images.size(0)
            running_loss += loss.item() * bsz
            num_samples += bsz

            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / max(num_samples, 1)
    preds = np.concatenate(all_preds, axis=0) if all_preds else np.array([])
    gt = np.concatenate(all_labels, axis=0) if all_labels else np.array([])
    mae = float(np.mean(np.abs(preds - gt))) if preds.size else np.nan

    return avg_loss, mae, preds, gt


def train_fold(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, device: torch.device, fold_idx: int, num_epochs: int = NUM_EPOCHS) -> Dict:
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=ADAMW_BETAS,
        weight_decay=WEIGHT_DECAY,
    )

    if LOSS_NAME.lower() == 'mse':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f'Unsupported LOSS_NAME: {LOSS_NAME}')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

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

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae, _, _ = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)

        scheduler.step()

        if (epoch + 1) % LOG_FREQUENCY == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d}/{num_epochs} | Train MSE: {train_loss:8.3f} | Val MSE: {val_loss:8.3f} | Val MAE: {val_mae:6.2f} cm | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val_mse:
            best_val_mse = val_loss
            torch.save(model.state_dict(), best_model_path)

    history['best_val_loss'] = best_val_mse

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_mae, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

    history['test_loss'] = test_loss
    history['test_mae'] = test_mae
    history['test_predictions'] = test_preds
    history['test_labels'] = test_labels

    return history


def compute_metrics(predictions: np.ndarray, ground_truth: np.ndarray) -> Dict:
    errors = np.abs(predictions - ground_truth).flatten()
    return {
        'mae': np.mean(errors),
        'std': np.std(errors),
        'median_ae': np.median(errors),
        'max_error': np.max(errors),
        'min_error': np.min(errors),
        'rmse': np.sqrt(np.mean((predictions - ground_truth) ** 2)),
    }


def print_metrics(metrics: Dict, title: str = 'Metrics'):
    print(f"\n{title}:")
    print(f"  MAE:        {metrics['mae']:.2f} ± {metrics['std']:.2f} cm")
    print(f"  Median AE:  {metrics['median_ae']:.2f} cm")
    print(f"  RMSE:       {metrics['rmse']:.2f} cm")
    print(f"  Min Error:  {metrics['min_error']:.2f} cm")
    print(f"  Max Error:  {metrics['max_error']:.2f} cm")
