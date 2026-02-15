
import sys
import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from config import get_device, WEIGHTS_PATH, IMG_SIZE
from utils import prepare_dataset, create_fold_splits, get_fold_dataframes
from dataset import LocalizerDataset
from model import create_model
from Visualization import (
    visualize_dataset_samples,
    plot_training_history,
    plot_error_distribution,
    plot_scatter_predictions,
    GradCAM
)
from Inference import HeightPredictor


def visualize_dataset(num_samples=5, data_subset='train'):
    """
    Visualize random samples from the dataset.

    Args:
        num_samples: Number of samples to display
        data_subset: 'train', 'val', or 'test'
    """
    print(f"\n{'=' * 60}")
    print(f"Visualizing {num_samples} samples from {data_subset} set")
    print(f"{'=' * 60}\n")

    # Load data
    data_df = prepare_dataset()
    patient_groups, _ = create_fold_splits(data_df, num_folds=4)
    train_df, val_df, test_df = get_fold_dataframes(data_df, patient_groups, fold_idx=0)

    # Select subset
    if data_subset == 'train':
        df = train_df
    elif data_subset == 'val':
        df = val_df
    else:
        df = test_df

    # Create dataset
    dataset = LocalizerDataset(df, is_train=False)

    # Visualize
    visualize_dataset_samples(dataset, num_samples=num_samples, title_prefix=f"{data_subset.capitalize()}")

    print(f"\n✓ Visualization complete!")
    print(f"  Dataset size: {len(dataset)} images")
    print(f"  Patients: {df['Patient_ID'].nunique()}")
    print(f"  Height range: {df['height_cm'].min():.1f} - {df['height_cm'].max():.1f} cm")


def visualize_predictions(model_path, fold_idx=0, num_samples=10):
    """
    Visualize model predictions vs ground truth.

    Args:
        model_path: Path to trained model checkpoint
        fold_idx: Which fold to visualize (0-3)
        num_samples: Number of samples to show
    """
    print(f"\n{'=' * 60}")
    print(f"Visualizing predictions from fold {fold_idx + 1}")
    print(f"{'=' * 60}\n")

    device = get_device()

    # Load data
    data_df = prepare_dataset()
    patient_groups, _ = create_fold_splits(data_df, num_folds=4)
    _, _, test_df = get_fold_dataframes(data_df, patient_groups, fold_idx=fold_idx)

    # Create dataset and loader
    test_dataset = LocalizerDataset(test_df, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Load model
    print("Loading model...")
    predictor = HeightPredictor(model_path=model_path)

    # Get predictions
    print("Making predictions...")
    results = predictor.predict_batch(test_loader, return_details=True)

    # Collect images for visualization
    all_images = []
    for images, _, _ in test_loader:
        all_images.append(images.cpu().numpy())
    all_images = np.concatenate(all_images, axis=0)

    # Plot predictions
    from Visualization import visualize_predictions
    visualize_predictions(
        images=all_images,
        predictions=results['predictions'],
        ground_truth=results['labels'],
        num_samples=num_samples
    )

    # Print statistics
    errors = np.abs(results['predictions'] - results['labels'])
    print(f"\n✓ Prediction statistics:")
    print(f"  Mean error: {errors.mean():.2f} cm")
    print(f"  Median error: {np.median(errors):.2f} cm")
    print(f"  Max error: {errors.max():.2f} cm")
    print(f"  Min error: {errors.min():.2f} cm")

    # Plot error distribution
    plot_error_distribution(errors)

    # Plot scatter
    plot_scatter_predictions(
        results['predictions'].reshape(-1, 1),
        results['labels'].reshape(-1, 1),
        title=f"Predictions vs Ground Truth (Fold {fold_idx + 1})"
    )


def visualize_gradcam(model_path, num_samples=3, fold_idx=0):
    """
    Visualize GradCAM attention maps.

    Args:
        model_path: Path to trained model checkpoint
        num_samples: Number of samples to visualize
        fold_idx: Which fold's test set to use
    """
    print(f"\n{'=' * 60}")
    print(f"Generating GradCAM visualizations")
    print(f"{'=' * 60}\n")

    import matplotlib.pyplot as plt

    device = get_device()

    # Load data
    data_df = prepare_dataset()
    patient_groups, _ = create_fold_splits(data_df, num_folds=4)
    _, _, test_df = get_fold_dataframes(data_df, patient_groups, fold_idx=fold_idx)

    # Create dataset
    test_dataset = LocalizerDataset(test_df, is_train=False)

    # Load model
    print("Loading model...")
    model = create_model(weights_path=str(WEIGHTS_PATH), device=str(device))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Create GradCAM
    gradcam = GradCAM(model, target_layer=model.backbone.layer4)

    # Visualize samples
    for i in range(min(num_samples, len(test_dataset))):
        print(f"\nGenerating GradCAM for sample {i + 1}/{num_samples}...")

        image, spacing, label = test_dataset[i]

        # Prepare for model
        image_batch = image.unsqueeze(0).to(device)
        spacing_batch = spacing.unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            prediction = model(image_batch, spacing_batch).item()

        # Generate GradCAM
        heatmap, overlay = gradcam.visualize(image_batch, spacing_batch)

        # Original image
        original = image[0].cpu().numpy()

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original, cmap='gray')
        axes[0].set_title(f'Original Image\nTrue: {label:.1f} cm')
        axes[0].axis('off')

        axes[1].imshow(heatmap)
        axes[1].set_title('GradCAM Heatmap\n(Model Attention)')
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title(f'Overlay\nPredicted: {prediction:.1f} cm\nError: {abs(prediction - label):.1f} cm')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(f'gradcam_sample_{i + 1}.png', dpi=150, bbox_inches='tight')
        print(f"  Saved to: gradcam_sample_{i + 1}.png")
        plt.show()

    print(f"\n✓ GradCAM visualizations complete!")


def visualize_training_history(results_excel='training_results_rotating.xlsx'):
    """
    Visualize training history from Excel results.

    Args:
        results_excel: Path to training results Excel file
    """
    print(f"\n{'=' * 60}")
    print(f"Visualizing training history")
    print(f"{'=' * 60}\n")

    import matplotlib.pyplot as plt

    # Load results
    results_df = pd.read_excel(results_excel, sheet_name='Detailed_Logs')

    # Get unique folds (excluding TEST_FINAL rows)
    folds = results_df[results_df['Epoch'] != 'TEST_FINAL']['Fold'].unique()

    # Plot each fold
    for fold in folds:
        fold_data = results_df[
            (results_df['Fold'] == fold) & (results_df['Epoch'] != 'TEST_FINAL')
            ]

        history = {
            'train_loss': fold_data['Train_MAE'].values,
            'val_loss': fold_data['Val_MAE'].values
        }

        plot_training_history(history, fold_idx=int(fold))

    # Plot summary across all folds
    summary_df = pd.read_excel(results_excel, sheet_name='Summary')

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    folds = summary_df['Fold'].values
    test_mae = summary_df['Test_MAE'].values

    ax.bar(folds, test_mae, color='steelblue', alpha=0.7)
    ax.axhline(test_mae.mean(), color='red', linestyle='--',
               label=f'Mean: {test_mae.mean():.2f} cm')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Test MAE (cm)')
    ax.set_title('Test Performance Across Folds')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cross_validation_summary.png', dpi=150, bbox_inches='tight')
    print(f"Saved summary to: cross_validation_summary.png")
    plt.show()

    print(f"\n✓ Training history visualization complete!")


def main():
    parser = argparse.ArgumentParser(description='Visualization tools for height prediction')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['dataset', 'predictions', 'gradcam', 'history'],
                        help='Visualization mode')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint (required for predictions/gradcam)')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold index (0-3)')
    parser.add_argument('--subset', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Data subset for dataset visualization')
    parser.add_argument('--results', type=str, default='training_results_rotating.xlsx',
                        help='Path to training results Excel file')

    args = parser.parse_args()

    if args.mode == 'dataset':
        visualize_dataset(num_samples=args.num_samples, data_subset=args.subset)

    elif args.mode == 'predictions':
        if args.model is None:
            print("Error: --model is required for predictions visualization")
            sys.exit(1)
        visualize_predictions(args.model, fold_idx=args.fold, num_samples=args.num_samples)

    elif args.mode == 'gradcam':
        if args.model is None:
            print("Error: --model is required for GradCAM visualization")
            sys.exit(1)
        visualize_gradcam(args.model, num_samples=args.num_samples, fold_idx=args.fold)

    elif args.mode == 'history':
        visualize_training_history(results_excel=args.results)


if __name__ == "__main__":
    # If run without arguments, show interactive menu
    if len(sys.argv) == 1:
        print("\n" + "=" * 60)
        print("VISUALIZATION TOOLS - Interactive Mode")
        print("=" * 60)
        print("\nAvailable visualizations:")
        print("  1. Dataset Samples - View preprocessed images")
        print("  2. Model Predictions - Compare predictions vs ground truth")
        print("  3. GradCAM - Model attention/interpretability")
        print("  4. Training History - Loss curves and performance")
        print("\n" + "=" * 60)

        choice = input("\nSelect option (1-4): ").strip()

        if choice == '1':
            num = input("Number of samples (default 5): ").strip()
            num = int(num) if num else 5
            subset = input("Subset (train/val/test, default train): ").strip()
            subset = subset if subset in ['train', 'val', 'test'] else 'train'
            visualize_dataset(num_samples=num, data_subset=subset)

        elif choice == '2':
            model_path = input("Model checkpoint path: ").strip()
            if not model_path:
                print("Error: Model path required")
                sys.exit(1)
            num = input("Number of samples (default 10): ").strip()
            num = int(num) if num else 10
            fold = input("Fold index 0-3 (default 0): ").strip()
            fold = int(fold) if fold else 0
            visualize_predictions(model_path, fold_idx=fold, num_samples=num)

        elif choice == '3':
            model_path = input("Model checkpoint path: ").strip()
            if not model_path:
                print("Error: Model path required")
                sys.exit(1)
            num = input("Number of samples (default 3): ").strip()
            num = int(num) if num else 3
            visualize_gradcam(model_path, num_samples=num)

        elif choice == '4':
            results = input("Results Excel path (default training_results_rotating.xlsx): ").strip()
            results = results if results else 'training_results_rotating.xlsx'
            visualize_training_history(results_excel=results)

        else:
            print("Invalid choice!")

    else:
        # Run with command line arguments
        main()