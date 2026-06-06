import cv2
import numpy as np
import nibabel as nib
import torch
import random
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
import os

from config import (
    IMG_SIZE, TARGET_PIXEL_SPACING_MM, WIN_MIN, WIN_MAX, THRESHOLD_VALUE,
    AUG_HORIZONTAL_FLIP_PROB, AUG_SHIFT_LIMIT, AUG_SCALE_LIMIT,
    AUG_ROTATE_LIMIT, AUG_SHIFT_SCALE_ROTATE_PROB,
    AUG_BRIGHTNESS_CONTRAST_PROB
)

# EfficientNet ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class LocalizerDataset(Dataset):
    """
    Dataset for loading and preprocessing CT Localizer images.

    Features:
    - Global Z-score Normalization
    - Robust Orientation (Standardizes vertical alignment)
    - Random Vertical Cropping (Simulates partial scans)
    - Dynamic Padding & Adaptive Intensity Handling
    - Physically Accurate Cropping and Resizing (Preserves accurate mm/px for model)
    """

    def __init__(self, df, is_train=False):
        self.df = df
        self.is_train = is_train

        if self.is_train:
            self.transform = A.Compose([
                # --- Geometric Augmentations ---
                A.HorizontalFlip(p=AUG_HORIZONTAL_FLIP_PROB),
                A.Affine(
                    translate_percent={
                        "x": (-AUG_SHIFT_LIMIT, AUG_SHIFT_LIMIT),
                        "y": (-AUG_SHIFT_LIMIT, AUG_SHIFT_LIMIT),
                    },
                    scale=1.0,
                    rotate=(-AUG_ROTATE_LIMIT, AUG_ROTATE_LIMIT),
                    fill=0,
                    p=AUG_SHIFT_SCALE_ROTATE_PROB
                ),
                # --- Intensity Augmentations ---
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=AUG_BRIGHTNESS_CONTRAST_PROB
                ),
                A.GaussNoise(std_range=(0.001, 0.005), p=0.3),
                # --- Normalization ---
                A.Normalize(
                    mean=IMAGENET_MEAN,
                    std=IMAGENET_STD,
                    max_pixel_value=1.0
                ),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(
                    mean=IMAGENET_MEAN,
                    std=IMAGENET_STD,
                    max_pixel_value=1.0
                ),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.df)

    def get_background_value(self, img):
        """Detects if background is Air (-1024) or Black (0)."""
        min_val = np.min(img)
        if min_val < -900:
            return -1024
        else:
            return min_val

    def load_json_metadata(self, nifti_path):
        """Finds and loads the corresponding JSON sidecar file."""
        json_path = nifti_path.replace('.nii.gz', '.json').replace('.nii', '.json')

        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                return json.load(f)
        return {}  # Return empty dict if missing

    def orient_from_metadata(self, img_2d, metadata):
        """
        Determines rotation and flipping based strictly on DICOM tags.
        Replaces the old threshold-based standardize_orientation.
        """
        rotated = False

        # 1. Default to standard image if metadata is missing
        if not metadata:
            return img_2d, False

        iop = metadata.get("ImageOrientationPatientDICOM", [1, 0, 0, 0, 1, 0])
        position = metadata.get("PatientPosition", "HFS")  # HFS = Head First Supine

        # 2. Extract Column Vector (how the image draws from top-to-bottom)
        c_x, c_y, c_z = iop[3], iop[4], iop[5]

        # 3. Check if image is Horizontal
        # If the magnitude of X or Y is greater than Z, the spine is drawn sideways.
        if abs(c_x) > abs(c_z) or abs(c_y) > abs(c_z):
            # The image is horizontal. Rotate 90 degrees.
            img_2d = cv2.rotate(img_2d, cv2.ROTATE_90_CLOCKWISE)
            rotated = True

        # 4. Handle Patient Position (HFS vs FFS)
        if position.startswith("FFS"):
            # Flip upside down if Feet First (Optional: Verify this visually on your dataset)
            img_2d = cv2.flip(img_2d, 0)  # 0 = vertical flip

        return img_2d, rotated

    def trim_empty_vertical_space(self, img):
        """Removes empty air above head and below feet."""
        img_u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, thresh = cv2.threshold(img_u8, 50, 255, cv2.THRESH_BINARY)

        row_sums = np.sum(thresh, axis=1)
        non_zero_rows = np.where(row_sums > 0)[0]

        if len(non_zero_rows) > 0:
            y_top = non_zero_rows[0]
            y_bottom = non_zero_rows[-1]
            return img[y_top:y_bottom, :]
        return img

    def random_vertical_crop(self, img):
        """Randomly crops height to simulate partial scans."""
        if not self.is_train:
            return img

        h, w = img.shape
        min_fraction = 0.6
        max_fraction = 1.0

        crop_ratio = random.uniform(min_fraction, max_fraction)
        new_h = int(h * crop_ratio)

        max_y_start = h - new_h
        y_start = random.randint(0, max_y_start)
        y_end = y_start + new_h

        return img[y_start:y_end, :]

    def resample_to_target_spacing(self, img, spacing, target_spacing_mm=TARGET_PIXEL_SPACING_MM):
        """Resample image to isotropic target pixel spacing before network resizing.

        Spacing is used here only to normalize the physical scale of the image; it is
        not passed to the model.
        """
        sx, sy = float(spacing[0]), float(spacing[1])
        h, w = img.shape

        new_w = max(1, int(round(w * (sx / target_spacing_mm))))
        new_h = max(1, int(round(h * (sy / target_spacing_mm))))

        resampled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resampled

    def resize_pad_dynamic(self, img, target_size):
        """Resizes and pads to a square target using the image-specific background value."""
        pad_value = self.get_background_value(img)

        h, w = img.shape

        scale = target_size / max(h, w)

        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        final = np.full((target_size, target_size), pad_value, dtype=np.float32)

        y_offset = (target_size - new_h) // 2

        x_offset = (target_size - new_w) // 2

        final[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return final

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        nifti_path = row['nifti_path']
        height_label = float(row['height_cm'])

        try:
            # 1. Load NIfTI
            nii = nib.load(nifti_path)
            img_data = nii.get_fdata()
            header = nii.header
            json_meta = self.load_json_metadata(nifti_path)

            if img_data.ndim >= 3:
                img_data = img_data.squeeze() if img_data.shape[-1] == 1 else np.max(img_data, axis=-1)

            if img_data.ndim > 2:
                img_data = img_data[..., img_data.shape[-1] // 2]
            elif img_data.ndim < 2:
                img_data = np.zeros((IMG_SIZE, IMG_SIZE))

            spacing = header.get_zooms()[:2]

            # 2. Windowing
            img_data = np.clip(img_data, WIN_MIN, WIN_MAX)
            img_data = (img_data - WIN_MIN) / (WIN_MAX - WIN_MIN)

            # 3. Orientation
            img_data, rotated = self.orient_from_metadata(img_data, json_meta)
            if rotated:
                spacing = (spacing[1], spacing[0])

            # 4. Trim Empty Space (Before cropping, so we crop the actual body)
            #img_data = self.trim_empty_vertical_space(img_data)

            # 5. Random Vertical Crop (Augmentation)
            img_data = self.random_vertical_crop(img_data)

            # 7. Resample to 1 mm spacing, then resize/pad for model input
            img_data = self.resample_to_target_spacing(img_data, spacing)
            img_data = self.resize_pad_dynamic(img_data, IMG_SIZE)

            # 8. Augmentations (includes Global Normalization)
            img_data = img_data.astype(np.float32)
            img_data = np.repeat(img_data[:, :, np.newaxis], 3, axis=2)
            augmented = self.transform(image=img_data)
            img_tensor = augmented['image']

            return img_tensor, torch.tensor(height_label, dtype=torch.float32)

        except Exception as e:
            print(f"Error loading {nifti_path}: {e}")
            return (torch.zeros((3, IMG_SIZE, IMG_SIZE)), torch.tensor(0.0))