
import torch
import pandas as pd
from pathlib import Path
from typing import List, Union, Dict
import numpy as np

from config import get_device, WEIGHTS_PATH, MODEL_CHECKPOINT_PATTERN
from dataset import LocalizerDataset
from model import create_model
from torch.utils.data import DataLoader


class HeightPredictor:
    """
    Class for loading trained models and making predictions.
    """

    def __init__(
            self,
            model_path: str = None,
            weights_path: str = None,
            device: str = None
    ):
        """
        Initialize predictor.

        Args:
            model_path: Path to trained model checkpoint (.pth file)
            weights_path: Path to RadImageNet pretrained weights
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if device else get_device())

        # Load model
        if weights_path is None:
            weights_path = str(WEIGHTS_PATH)

        self.model = create_model(weights_path=weights_path, device=str(self.device))

        # Load trained checkpoint if provided
        if model_path:
            self.load_checkpoint(model_path)

        self.model.eval()

    def load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint"""
        print(f"Loading checkpoint from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print("✓ Checkpoint loaded successfully")

    def predict_single(
            self,
            image_tensor: torch.Tensor,
            spacing_tensor: torch.Tensor
    ) -> float:
        """
        Predict height for a single image.

        Args:
            image_tensor: Image tensor (C, H, W) or (1, C, H, W)
            spacing_tensor: Spacing tensor (2,) or (1, 2)

        Returns:
            Predicted height in cm
        """
        # Ensure batch dimension
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if spacing_tensor.ndim == 1:
            spacing_tensor = spacing_tensor.unsqueeze(0)

        # Move to device
        image_tensor = image_tensor.to(self.device)
        spacing_tensor = spacing_tensor.to(self.device)

        # Predict
        with torch.no_grad():
            prediction = self.model(image_tensor, spacing_tensor)

        return prediction.item()

    def predict_batch(
            self,
            dataloader: DataLoader,
            return_details: bool = False
    ) -> Union[np.ndarray, Dict]:
        """
        Predict heights for a batch of images.

        Args:
            dataloader: DataLoader with images to predict
            return_details: If True, return dict with predictions and metadata

        Returns:
            Array of predictions or dict with details
        """
        all_predictions = []
        all_spacings = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                images, spacings, labels = batch
                images = images.to(self.device)
                spacings = spacings.to(self.device)

                predictions = self.model(images, spacings)

                all_predictions.append(predictions.cpu().numpy())
                all_spacings.append(spacings.cpu().numpy())
                all_labels.append(labels.numpy())

        predictions = np.concatenate(all_predictions, axis=0)

        if return_details:
            return {
                'predictions': predictions.flatten(),
                'spacings': np.concatenate(all_spacings, axis=0),
                'labels': np.concatenate(all_labels, axis=0),
            }

        return predictions.flatten()

    def predict_from_dataframe(
            self,
            df: pd.DataFrame,
            batch_size: int = 8
    ) -> pd.DataFrame:
        """
        Predict heights for all samples in a DataFrame.

        Args:
            df: DataFrame with columns ['Patient_ID', 'nifti_path', 'height_cm']
            batch_size: Batch size for inference

        Returns:
            DataFrame with predictions added
        """
        # Create dataset and loader
        dataset = LocalizerDataset(df, is_train=False)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Get predictions
        results = self.predict_batch(loader, return_details=True)

        # Add to dataframe
        df_copy = df.copy()
        df_copy['predicted_height'] = results['predictions']
        df_copy['error'] = np.abs(df_copy['predicted_height'] - df_copy['height_cm'])
        df_copy['spacing_x'] = results['spacings'][:, 0]
        df_copy['spacing_y'] = results['spacings'][:, 1]

        return df_copy


def ensemble_predict(
        model_paths: List[str],
        dataloader: DataLoader,
        weights_path: str = None,
        device: str = None
) -> np.ndarray:
    """
    Make ensemble predictions from multiple model checkpoints.

    Args:
        model_paths: List of paths to model checkpoints
        dataloader: DataLoader with data to predict
        weights_path: Path to RadImageNet weights
        device: Device to run on

    Returns:
        Array of ensemble predictions (mean of all models)
    """
    all_model_predictions = []

    for model_path in model_paths:
        print(f"Loading model: {model_path}")
        predictor = HeightPredictor(
            model_path=model_path,
            weights_path=weights_path,
            device=device
        )
        predictions = predictor.predict_batch(dataloader)
        all_model_predictions.append(predictions)

    # Average predictions
    ensemble_predictions = np.mean(all_model_predictions, axis=0)

    return ensemble_predictions


def predict_from_excel(
        excel_path: str,
        model_path: str,
        output_path: str = None,
        batch_size: int = 8
) -> pd.DataFrame:
    """
    Convenience function to predict heights from an Excel file.

    Args:
        excel_path: Path to Excel file with patient data
        model_path: Path to trained model checkpoint
        output_path: Path to save results (optional)
        batch_size: Batch size for inference

    Returns:
        DataFrame with predictions
    """
    from utils import prepare_dataset

    # Temporarily override excel path
    import config
    original_path = config.EXCEL_PATH
    config.EXCEL_PATH = Path(excel_path)

    # Load data
    df = prepare_dataset()

    # Restore original path
    config.EXCEL_PATH = original_path

    # Make predictions
    predictor = HeightPredictor(model_path=model_path)
    results_df = predictor.predict_from_dataframe(df, batch_size=batch_size)

    # Save if requested
    if output_path:
        results_df.to_excel(output_path, index=False)
        print(f"Results saved to: {output_path}")

    # Print summary
    print("\nPrediction Summary:")
    print(f"  Total samples: {len(results_df)}")
    print(f"  Mean error: {results_df['error'].mean():.2f} cm")
    print(f"  Median error: {results_df['error'].median():.2f} cm")
    print(f"  Max error: {results_df['error'].max():.2f} cm")

    return results_df


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Predict heights from CT localizers')
    parser.add_argument('--excel', type=str, required=True, help='Path to Excel file')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='predictions.xlsx', help='Output path')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')

    args = parser.parse_args()

    # Run prediction
    results = predict_from_excel(
        excel_path=args.excel,
        model_path=args.model,
        output_path=args.output,
        batch_size=args.batch_size
    )

    print("\nTop 10 predictions:")
    print(results[['Patient_ID', 'height_cm', 'predicted_height', 'error']].head(10))