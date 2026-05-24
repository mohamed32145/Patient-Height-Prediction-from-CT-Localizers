
import numpy as np
import torch
from monai.data import DataLoader

from config import (
    NUM_FOLDS, BATCH_SIZE, RESULTS_EXCEL_PATH, DROPOUT_RATE,
    setup_directories, get_device, EXPERIMENTS_DIR,
    FORCED_TEST_PATIENTS_BY_FOLD, FORCED_VAL_PATIENTS_BY_FOLD,
)
from utils import (
    prepare_dataset,
    create_fold_splits_train_val_test,
    get_fold_dataframes_explicit,
    save_results_to_excel,
    save_fold_predictions,
)
from dataset_monai import LocalizerDatasetMONAI
from model import create_model
from Train import train_fold, compute_metrics, print_metrics


def main():
    """
    Main training pipeline with rotating cross-validation.

    This implements a 4-fold rotating CV where:
    - Group i         -> TEST (Held out completely)
    - Group (i+1)%4   -> VALIDATION (Used for model tuning)
    - Remaining 2     -> TRAIN
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
        num_folds=NUM_FOLDS,
        forced_test_patients_by_fold=FORCED_TEST_PATIENTS_BY_FOLD,
        forced_val_patients_by_fold=FORCED_VAL_PATIENTS_BY_FOLD,  # optional; can omit to auto-derive
        test_frac=0.25,
        val_frac=0.20,
        random_seed=42
    )

    # ========================================================================
    # 3. ROTATING CROSS-VALIDATION LOOP
    # ========================================================================
    print("\nStep 3: Training with rotating cross-validation...")
    print("-" * 80)

    fold_performance = []
    all_results = []
    all_histories = []

    for fold_idx in range(NUM_FOLDS):
        # Get train/val/test splits for this fold

        train_df, val_df, test_df = get_fold_dataframes_explicit(
            data_df=data_df,
            test_groups=test_groups,
            val_groups=val_groups,
            train_groups=train_groups,
            fold_idx=fold_idx
        )

        # Create datasets — MONAI dict-based pipeline
        train_dataset = LocalizerDatasetMONAI(train_df, is_train=True)
        val_dataset = LocalizerDatasetMONAI(val_df, is_train=False)
        test_dataset = LocalizerDatasetMONAI(test_df, is_train=False)

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
        save_fold_predictions(test_df, history['test_predictions'], fold_idx, str(EXPERIMENTS_DIR))
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

    # Save results to Excel
    print(f"\nSaving results to {RESULTS_EXCEL_PATH}...")
    save_results_to_excel(all_results, fold_performance, RESULTS_EXCEL_PATH)

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