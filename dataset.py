
import cv2
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import (
    IMG_SIZE, WIN_MIN, WIN_MAX, CROP_WIDTH, THRESHOLD_VALUE,
    AUG_HORIZONTAL_FLIP_PROB, AUG_SHIFT_LIMIT, AUG_SCALE_LIMIT,
    AUG_ROTATE_LIMIT, AUG_SHIFT_SCALE_ROTATE_PROB,
    AUG_BRIGHTNESS_CONTRAST_PROB
)


class LocalizerDataset(Dataset):
    """
    Dataset for loading and preprocessing CT Localizer images.

    Features:
    - Loads NIfTI files
    - Applies bone windowing
    - Standardizes orientation (vertical body)
    - Crops to spine region
    - Resizes with padding
    - Applies augmentations for training
    """

    def __init__(self, df, is_train=False):
        """
        Args:
            df: DataFrame with columns ['Patient_ID', 'nifti_path', 'height_cm']
            is_train: Whether to apply training augmentations
        """
        self.df = df
        self.is_train = is_train

        # Define augmentation pipeline
        if self.is_train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=AUG_HORIZONTAL_FLIP_PROB),
                A.ShiftScaleRotate(
                    shift_limit=AUG_SHIFT_LIMIT,
                    scale_limit=AUG_SCALE_LIMIT,
                    rotate_limit=AUG_ROTATE_LIMIT,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=AUG_SHIFT_SCALE_ROTATE_PROB
                ),
                A.RandomBrightnessContrast(p=AUG_BRIGHTNESS_CONTRAST_PROB),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.df)

    def standardize_orientation(self, img_2d):
        """
        Detects if the body is horizontal and rotates it to vertical.

        Args:
            img_2d: 2D numpy array of the image

        Returns:
            Tuple of (rotated_image, was_rotated)
        """
        img_u8 = cv2.normalize(img_2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Threshold to find body (ignores bed lines)
        _, thresh = cv2.threshold(img_u8, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img_2d, False

        # Find largest contour (body)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # If wider than tall, rotate 90 degrees
        if w > h:
            img_rotated = cv2.rotate(img_2d, cv2.ROTATE_90_CLOCKWISE)
            return img_rotated, True

        return img_2d, False

    def crop_to_spine(self, img_2d):
        """
        Centers on the spine and takes a narrow crop to exclude bed rails.

        Args:
            img_2d: 2D numpy array of the image

        Returns:
            Cropped image centered on spine
        """
        img_u8 = cv2.normalize(img_2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, thresh = cv2.threshold(img_u8, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

        # Find center of mass (spine location)
        M = cv2.moments(thresh)
        if M["m00"] == 0:
            center_x = img_2d.shape[1] // 2
        else:
            center_x = int(M["m10"] / M["m00"])

        # Find vertical extent
        row_sums = np.sum(thresh, axis=1)
        non_zero_rows = np.where(row_sums > 0)[0]
        if len(non_zero_rows) > 0:
            y_top, y_bottom = non_zero_rows[0], non_zero_rows[-1]
        else:
            y_top, y_bottom = 0, img_2d.shape[0]

        # Crop horizontally around center
        h, w = img_2d.shape
        x_start = max(0, center_x - CROP_WIDTH // 2)
        x_end = min(w, center_x + CROP_WIDTH // 2)

        return img_2d[y_top:y_bottom, x_start:x_end]

    def resize_pad(self, img):
        """
        Resize image maintaining aspect ratio and pad to square.

        Args:
            img: 2D numpy array

        Returns:
            Resized and padded image of size (IMG_SIZE, IMG_SIZE)
        """
        h, w = img.shape
        scale = IMG_SIZE / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded output
        final_img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        y_offset = (IMG_SIZE - new_h) // 2
        x_offset = (IMG_SIZE - new_w) // 2
        final_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img

        return final_img

    def __getitem__(self, idx):
        """
        Load and preprocess a single sample.

        Returns:
            Tuple of (image_tensor, spacing_tensor, height_label)
        """
        row = self.df.iloc[idx]
        nifti_path = row['nifti_path']
        height_label = float(row['height_cm'])

        try:
            # 1. Load NIfTI
            nii = nib.load(nifti_path)
            img_data = nii.get_fdata()
            header = nii.header

            # Handle 3D volumes - extract 2D slice
            if img_data.ndim >= 3:
                if img_data.shape[-1] == 1:
                    img_data = img_data.squeeze()
                else:
                    img_data = np.max(img_data, axis=-1)

            if img_data.ndim != 2:
                if img_data.ndim > 2:
                    img_data = img_data[..., img_data.shape[-1] // 2]
                else:
                    img_data = np.squeeze(img_data)

            # 2. Extract pixel spacing metadata
            spacing = header.get_zooms()[:2]

            # 3. Apply bone windowing and normalize to [0, 1]
            img_data = np.clip(img_data, WIN_MIN, WIN_MAX)
            img_data = (img_data - WIN_MIN) / (WIN_MAX - WIN_MIN)

            # 4. Standardize orientation (rotate horizontal bodies to vertical)
            img_data, rotated = self.standardize_orientation(img_data)

            # Important: If rotated, swap pixel spacing dimensions
            if rotated:
                spacing = (spacing[1], spacing[0])

            spacing = torch.tensor(spacing, dtype=torch.float32)

            # 5. Crop to spine region (now that body is vertical)
            img_data = self.crop_to_spine(img_data)

            # 6. Resize with padding
            img_data = self.resize_pad(img_data)

            # 7. Apply augmentations
            img_data = img_data.astype(np.float32)[:, :, np.newaxis]
            augmented = self.transform(image=img_data)
            img_tensor = augmented['image']

            # Convert grayscale to 3-channel (for ResNet)
            img_tensor = img_tensor.repeat(3, 1, 1)

            return img_tensor, spacing, torch.tensor(height_label, dtype=torch.float32)

        except Exception as e:
            print(f"Error loading {nifti_path}: {e}")
            # Return zero tensors on error
            return (
                torch.zeros((3, IMG_SIZE, IMG_SIZE)),
                torch.zeros(2),
                torch.tensor(0.0)
            )