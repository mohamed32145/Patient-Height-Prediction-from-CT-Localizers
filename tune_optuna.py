import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    NUM_EPOCHS,
    NUM_FOLDS,
    RANDOM_SEED,
    setup_directories,
    get_device,
)
from dataset import LocalizerDataset
from model import create_model
from Train import train_fold
from utils import prepare_dataset, create_fold_splits, get_fold_dataframes


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_loader(ds, batch_size, shuffle, device):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False,
    )


def run_cv_trial(params, data_df, patient_groups, device, epochs, trial=None):
    fold_val_maes = []
    fold_test_maes = []

    for fold_idx in range(NUM_FOLDS):
        train_df, val_df, test_df = get_fold_dataframes(data_df, patient_groups, fold_idx)

        train_dataset = LocalizerDataset(
            train_df,
            is_train=True,
            rotate_limit=params['rotate_limit'],
            brightness_contrast_prob=params['brightness_contrast_prob'],
        )
        val_dataset = LocalizerDataset(val_df, is_train=False)
        test_dataset = LocalizerDataset(test_df, is_train=False)

        train_loader = _make_loader(train_dataset, params['batch_size'], True, device)
        val_loader = _make_loader(val_dataset, params['batch_size'], False, device)
        test_loader = _make_loader(test_dataset, params['batch_size'], False, device)

        model = create_model(
            device=str(device),
            dropout_rate=params['dropout'],
            init_mode=params['init_mode'],
        )

        history = train_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            fold_idx=fold_idx,
            num_epochs=epochs,
            learning_rate=params['lr'],
            weight_decay=params['weight_decay'],
            cosine_eta_min=params['cosine_eta_min'],
            use_cosine_annealing=params['use_cosine_annealing'],
        )

        fold_val_maes.append(float(np.min(history['val_mae'])))
        fold_test_maes.append(float(history['test_mae']))

        if trial is not None:
            running_mean = float(np.mean(fold_val_maes))
            trial.report(running_mean, step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return {
        'fold_val_maes': fold_val_maes,
        'fold_test_maes': fold_test_maes,
        'mean_val_mae': float(np.mean(fold_val_maes)),
        'std_val_mae': float(np.std(fold_val_maes)),
        'mean_test_mae': float(np.mean(fold_test_maes)),
    }


def build_objective(data_df, patient_groups, device, epochs, metadata_path):
    def objective(trial):
        start = time.time()

        # Minimal “good default” search space + init/freezing mode.
        params = {
            'lr': trial.suggest_categorical('lr', [3e-5, 1e-4, 2e-4]),
            'weight_decay': trial.suggest_categorical('weight_decay', [1e-6, 1e-5, 1e-4]),
            'dropout': trial.suggest_categorical('dropout', [0.2, 0.3, 0.4]),
            'batch_size': trial.suggest_categorical('batch_size', [8, 16]),
            'cosine_eta_min': trial.suggest_categorical('cosine_eta_min', [1e-7, 1e-6, 1e-5]),
            'rotate_limit': trial.suggest_categorical('rotate_limit', [5, 10, 15]),
            'brightness_contrast_prob': trial.suggest_categorical('brightness_contrast_prob', [0.1, 0.2, 0.3]),
            'init_mode': trial.suggest_int('init_mode', -1, 4),
            'use_cosine_annealing': True,
        }

        set_all_seeds(RANDOM_SEED)

        metrics = run_cv_trial(
            params=params,
            data_df=data_df,
            patient_groups=patient_groups,
            device=device,
            epochs=epochs,
            trial=trial,
        )

        elapsed_sec = time.time() - start
        trial.set_user_attr('fold_val_maes', metrics['fold_val_maes'])
        trial.set_user_attr('fold_test_maes', metrics['fold_test_maes'])
        trial.set_user_attr('std_val_mae', metrics['std_val_mae'])
        trial.set_user_attr('runtime_sec', elapsed_sec)
        trial.set_user_attr('epochs', epochs)

        row = {
            'trial': trial.number,
            'params': params,
            'fold_val_maes': metrics['fold_val_maes'],
            'fold_test_maes': metrics['fold_test_maes'],
            'mean_val_mae': metrics['mean_val_mae'],
            'std_val_mae': metrics['std_val_mae'],
            'runtime_sec': elapsed_sec,
            'epochs': epochs,
        }
        with metadata_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row) + '\n')

        return metrics['mean_val_mae']

    return objective


def main():
    parser = argparse.ArgumentParser(description='Optuna hyperparameter tuning for CT-localizer height regression')
    parser.add_argument('--phase', choices=['coarse', 'refine'], default='coarse')
    parser.add_argument('--trials', type=int, default=24)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--study-name', type=str, default='height_regression_optuna')
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_height.db')
    parser.add_argument('--output-dir', type=str, default='experiments_height_pytorch/optuna')
    args = parser.parse_args()

    setup_directories()
    device = get_device()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = out_dir / f'{args.study_name}_{args.phase}_trials.jsonl'

    print('Preparing dataset and fixed folds...')
    data_df = prepare_dataset()
    patient_groups, _ = create_fold_splits(data_df, num_folds=NUM_FOLDS)

    objective = build_objective(
        data_df=data_df,
        patient_groups=patient_groups,
        device=device,
        epochs=args.epochs,
        metadata_path=metadata_path,
    )

    study = optuna.create_study(
        direction='minimize',
        study_name=f'{args.study_name}_{args.phase}',
        storage=args.storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=4),
    )

    study.optimize(objective, n_trials=args.trials)

    print('\nBest trial summary:')
    print(f'  Trial: {study.best_trial.number}')
    print(f'  Mean CV Val MAE: {study.best_value:.4f}')
    print(f'  Params: {study.best_trial.params}')


if __name__ == '__main__':
    main()
