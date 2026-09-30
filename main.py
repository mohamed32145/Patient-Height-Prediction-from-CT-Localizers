
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from config import (
    NUM_FOLDS, BATCH_SIZE, RESULTS_EXCEL_PATH,DROPOUT_RATE,
    setup_directories, get_device,EXPERIMENTS_DIR
)
from utils import (
    prepare_dataset,
    create_fold_splits_train_val_test,
    summarize_fold_splits,
    get_fold_dataframes_explicit,
    compute_height_range_errors,
    print_height_range_errors,
    save_results_to_excel,
    save_fold_predictions
)
from dataset import LocalizerDataset
from model import create_model
from Train import train_fold, compute_metrics, print_metrics


def main():
    """
    Main training pipeline with height-stratified cross-validation.

    Patients are split into 4 folds with the same height distribution. In fold i:
    - Fold i                                      -> TEST (Held out completely)
    - VAL_FRAC of all patients, from other folds  -> VALIDATION (Used for model tuning)
    - Remaining patients                          -> TRAIN
    """

    # Setup
    print("\n" + "=" * 80)
    print("HEIGHT PREDICTION FROM CT LOCALIZERS - TRAINING PIPELINE")
    print("=" * 80 + "\n")

    setup_directories()
    device = get_device()
    print(f"Using device: {device}\n")

    # ========================================================================
    # 1. PREPARE DATASET
    # ========================================================================
    print("Step 1: Loading and preparing dataset...")
    print("-" * 80)
    data_df = prepare_dataset()

    # ========================================================================
    # 2. CREATE FOLD SPLITS
    # ========================================================================
    print("\nStep 2: Creating cross-validation splits...")
    print("-" * 80)

    test_groups, val_groups, train_groups, all_patient_ids = create_fold_splits_train_val_test(
        data_df=data_df,
        num_folds=NUM_FOLDS
    )

    split_summary = summarize_fold_splits(data_df, test_groups, val_groups, train_groups)
    print(split_summary.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    split_summary_path = EXPERIMENTS_DIR / 'cv_split_summary.csv'
    split_summary.round(2).to_csv(split_summary_path, index=False)
    print(f"  -> Saved split summary to {split_summary_path.name}")

    # ========================================================================
    # 3. CROSS-VALIDATION LOOP
    # ========================================================================
    print("\nStep 3: Training with height-stratified cross-validation...")
    print("-" * 80)

    fold_performance = []
    all_results = []
    all_histories = []
    all_test_predictions = []

    for fold_idx in range(NUM_FOLDS):
        # Get train/val/test splits for this fold

        train_df, val_df, test_df = get_fold_dataframes_explicit(
            data_df=data_df,
            test_groups=test_groups,
            val_groups=val_groups,
            train_groups=train_groups,
            fold_idx=fold_idx
        )

        # Create datasets
        train_dataset = LocalizerDataset(train_df, is_train=True)
        val_dataset = LocalizerDataset(val_df, is_train=False)
        test_dataset = LocalizerDataset(test_df, is_train=False)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True if device.type == 'cuda' else False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True if device.type == 'cuda' else False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True if device.type == 'cuda' else False
        )

        # Create fresh model for this fold
        model = create_model(device=str(device),dropout_rate=DROPOUT_RATE,init_mode=0)

        # Train the fold
        history = train_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            fold_idx=fold_idx
        )
        fold_predictions = save_fold_predictions(test_df, history['test_predictions'], fold_idx, str(EXPERIMENTS_DIR))
        all_test_predictions.append(fold_predictions)
        # Store results
        fold_performance.append(history['test_mae'])
        all_histories.append(history)

        # Log detailed results for each epoch
        for epoch_idx, (train_loss, val_loss) in enumerate(
                zip(history['train_loss'], history['val_loss'])
        ):
            all_results.append({
                'Fold': fold_idx + 1,
                'Epoch': epoch_idx + 1,
                'Train_MSE': train_loss,
                'Val_MSE': val_loss,
                'Val_MAE': history['val_mae'][epoch_idx],
                'Test_MAE': None
            })

        # Add final test result
        all_results.append({
            'Fold': fold_idx + 1,
            'Epoch': 'TEST_FINAL',
            'Train_MSE': None,
            'Val_MSE': history['best_val_loss'],
            'Val_MAE': None,
            'Test_MAE': history['test_mae']
        })

        # Compute and print metrics for this fold
        metrics = compute_metrics(
            history['test_predictions'],
            history['test_labels']
        )
        print_metrics(metrics, title=f"Fold {fold_idx + 1} Test Metrics")
        print_height_range_errors(
            compute_height_range_errors(fold_predictions),
            title=f"Fold {fold_idx + 1} Test Error by Height Range"
        )

    # ========================================================================
    # 4. SAVE RESULTS AND SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION COMPLETE")
    print("=" * 80)

    # Print summary statistics
    print(f"\nTest MAE across {NUM_FOLDS} folds:")
    for i, mae in enumerate(fold_performance):
        print(f"  Fold {i + 1}: {mae:.2f} cm")

    print(f"\nOverall Performance:")
    print(f"  Mean Test MAE: {np.mean(fold_performance):.2f} ± {np.std(fold_performance):.2f} cm")
    print(f"  Median Test MAE: {np.median(fold_performance):.2f} cm")
    print(f"  Min Test MAE: {np.min(fold_performance):.2f} cm")
    print(f"  Max Test MAE: {np.max(fold_performance):.2f} cm")

    # Each patient is in exactly one test fold, so pooling the folds scores the whole group once
    all_predictions = pd.concat(all_test_predictions, ignore_index=True)
    print(f"\nAll {all_predictions['Patient_ID'].nunique()} Patients Together "
          f"({len(all_predictions)} test images from the {NUM_FOLDS} folds):")
    print(f"  Test MAE: {all_predictions['Absolute_Error'].mean():.2f} cm")

    height_range_errors = compute_height_range_errors(all_predictions)
    print_height_range_errors(height_range_errors, title="Test Error by Height Range (all folds together)")

    # Save results to Excel
    print(f"\nSaving results to {RESULTS_EXCEL_PATH}...")
    save_results_to_excel(all_results, fold_performance, RESULTS_EXCEL_PATH, height_range_errors)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80 + "\n")

    return fold_performance, all_histories


if __name__ == "__main__":
    # Run the main training pipeline
    fold_performance, histories = main()

    # Optional: Print final summary
    print("\nFinal Summary:")
    print(f"  Average Test MAE: {np.mean(fold_performance):.2f} cm")
    print(f"  Best Fold: Fold {np.argmin(fold_performance) + 1} ({np.min(fold_performance):.2f} cm)")
    print(f"  Worst Fold: Fold {np.argmax(fold_performance) + 1} ({np.max(fold_performance):.2f} cm)")