import cv2
import numpy as np
import nibabel as nib
import torch
import random  # Added for random cropping
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import (
    IMG_SIZE, WIN_MIN, WIN_MAX, THRESHOLD_VALUE,
    AUG_HORIZONTAL_FLIP_PROB, AUG_SHIFT_LIMIT, AUG_SCALE_LIMIT,
    AUG_ROTATE_LIMIT, AUG_SHIFT_SCALE_ROTATE_PROB,
    AUG_BRIGHTNESS_CONTRAST_PROB
)

# --- GLOBAL STATS
GLOBAL_MEAN = 0.185
GLOBAL_STD = 0.265


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

    def __init__(self, df, is_train=False, mean=GLOBAL_MEAN, std=GLOBAL_STD):
        self.df = df
        self.is_train = is_train
        self.mean = mean
        self.std = std

        # Define augmentation pipeline
        if self.is_train:
            self.transform = A.Compose([
                # --- Geometric Augmentations ---
                A.HorizontalFlip(p=AUG_HORIZONTAL_FLIP_PROB),
                A.ShiftScaleRotate(
                    shift_limit=AUG_SHIFT_LIMIT,
                    # FIXED: scale_limit set to 0.0 to prevent artificial zooming that breaks physical mm/px mapping
                    scale_limit=0.0,
                    rotate_limit=AUG_ROTATE_LIMIT,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=AUG_SHIFT_SCALE_ROTATE_PROB
                ),

                # --- Intensity Augmentations ---
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=AUG_BRIGHTNESS_CONTRAST_PROB
                ),
                A.GaussNoise(var_limit=(0.001, 0.005), p=0.3),

                # --- Normalization (Dataset Level) ---
                A.Normalize(
                    mean=[self.mean],
                    std=[self.std],
                    max_pixel_value=1.0
                ),

                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(
                    mean=[self.mean],
                    std=[self.std],
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

    def standardize_orientation(self, img_2d):
        """Rotates image if width > height."""
        img_u8 = cv2.normalize(img_2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, thresh = cv2.threshold(img_u8, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img_2d, False

        largest_contour = max(contours, key=cv2.contourArea)
        _, _, w, h = cv2.boundingRect(largest_contour)

        if w > h:
            img_rotated = cv2.rotate(img_2d, cv2.ROTATE_90_CLOCKWISE)
            return img_rotated, True

        return img_2d, False

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

    def resize_pad_dynamic_with_spacing(self, img, spacing, target_size):

        """

        FIXED: Resizes and pads using image-specific background value, and

        recalculates the new pixel spacing so the model's metadata branch gets accurate data.

        """

        pad_value = self.get_background_value(img)

        h, w = img.shape

        scale = target_size / max(h, w)

        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        final = np.full((target_size, target_size), pad_value, dtype=np.float32)

        y_offset = (target_size - new_h) // 2

        x_offset = (target_size - new_w) // 2

        final[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        # Calculate the new spacing after resizing

        new_spacing_x = spacing[0] / scale

        new_spacing_y = spacing[1] / scale

        new_spacing = (new_spacing_x, new_spacing_y)

        return final, new_spacing

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        nifti_path = row['nifti_path']
        height_label = float(row['height_cm'])

        try:
            # 1. Load NIfTI
            nii = nib.load(nifti_path)
            img_data = nii.get_fdata()
            header = nii.header

            if img_data.ndim >= 3: img_data = img_data.squeeze() if img_data.shape[-1] == 1 else np.max(img_data,
                                                                                                        axis=-1)
            if img_data.ndim > 2:
                img_data = img_data[..., img_data.shape[-1] // 2]
            elif img_data.ndim < 2:
                img_data = np.zeros((IMG_SIZE, IMG_SIZE))

            spacing = header.get_zooms()[:2]

            # 2. Windowing
            img_data = np.clip(img_data, WIN_MIN, WIN_MAX)
            img_data = (img_data - WIN_MIN) / (WIN_MAX - WIN_MIN)

            # 3. Orientation
            img_data, rotated = self.standardize_orientation(img_data)
            if rotated: spacing = (spacing[1], spacing[0])

            # 4. Trim Empty Space (Before cropping, so we crop the actual body)
            img_data = self.trim_empty_vertical_space(img_data)

            # 5. Random Vertical Crop (Augmentation)
            img_data = self.random_vertical_crop(img_data)

            # 6. Crop to Spine (Passing spacing for physical accuracy)
            # img_data = self.crop_to_spine_physical(img_data, spacing)

            # 7. Resize & Pad (Capturing the updated spacing!)
            img_data, updated_spacing = self.resize_pad_dynamic_with_spacing(img_data, spacing, IMG_SIZE)

            # 8. Create spacing tensor using the accurately calculated numbers
            spacing_tensor = torch.tensor(updated_spacing, dtype=torch.float32)

            # 9. Augmentations (includes Global Normalization)
            img_data = img_data.astype(np.float32)[:, :, np.newaxis]
            augmented = self.transform(image=img_data)
            img_tensor = augmented['image']
            img_tensor = img_tensor.repeat(3, 1, 1)

            return img_tensor, spacing_tensor, torch.tensor(height_label, dtype=torch.float32)

        except Exception as e:
            print(f"Error loading {nifti_path}: {e}")
            return (torch.zeros((3, IMG_SIZE, IMG_SIZE)), torch.zeros(2), torch.tensor(0.0))

# class LocalizerDataset(Dataset):
#     """
#     Dataset for loading and preprocessing CT Localizer images.
#
#     UPDATES:
#     - Removed 'Physical Width Crop': Keeps original image width.
#     - Added 'Random Vertical Crop': Simulates partial spine scans (C-spine, L-spine, etc.).
#     """
#
#     def __init__(self, df, is_train=False):
#         self.df = df
#         self.is_train = is_train
#
#         # Standard augmentations (Geometric & Pixel-level)
#         if self.is_train:
#             self.transform = A.Compose([
#                 A.HorizontalFlip(p=AUG_HORIZONTAL_FLIP_PROB),
#                 A.ShiftScaleRotate(
#                     shift_limit=AUG_SHIFT_LIMIT,
#                     scale_limit=AUG_SCALE_LIMIT,
#                     rotate_limit=AUG_ROTATE_LIMIT,
#                     border_mode=cv2.BORDER_CONSTANT,
#                     value=0,
#                     p=AUG_SHIFT_SCALE_ROTATE_PROB
#                 ),
#                 A.RandomBrightnessContrast(p=AUG_BRIGHTNESS_CONTRAST_PROB),
#                 ToTensorV2()
#             ])
#         else:
#             self.transform = A.Compose([
#                 ToTensorV2()
#             ])
#
#     def __len__(self):
#         return len(self.df)
#
#     def get_background_value(self, img):
#         """Detects if background is Air (-1024) or Black (0)."""
#         min_val = np.min(img)
#         if min_val < -900:
#             return -1024
#         else:
#             return min_val
#
#     def standardize_orientation_robust(self, img):
#         """Rotates image if width > height."""
#         img_u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#         _, thresh = cv2.threshold(img_u8, 50, 255, cv2.THRESH_BINARY)
#         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#         if not contours: return img, False
#
#         largest = max(contours, key=cv2.contourArea)
#         _, _, w, h = cv2.boundingRect(largest)
#
#         if w > h:
#             return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), True
#         return img, False
#
#     # Random Vertical Crop ---
#     def random_vertical_crop(self, img):
#         """
#         Randomly crops the image vertically to simulate partial scans.
#         It keeps 100% of the width, but cuts the height.
#
#         Logic:
#         - If training: Randomly keep 60% to 100% of the height.
#         - If validation: Keep 100% (Full body).
#         """
#         if not self.is_train:
#             return img
#
#         h, w = img.shape
#
#         # Define minimum visible portion (e.g., 60% of the original image)
#         min_fraction = 0.6
#         max_fraction = 1.0
#
#         # Randomly choose a crop height
#         crop_ratio = random.uniform(min_fraction, max_fraction)
#         new_h = int(h * crop_ratio)
#
#         # Randomly choose a starting Y position
#         # If we crop 80%, we have 20% "slack" to slide the window up or down
#         max_y_start = h - new_h
#         y_start = random.randint(0, max_y_start)
#         y_end = y_start + new_h
#
#         return img[y_start:y_end, :]
#
#     def trim_empty_vertical_space(self, img):
#         """
#         Deterministic trim: Removes empty air above head and below feet.
#         This ensures our random crop works on the BODY, not the empty air.
#         """
#         img_u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#         _, thresh = cv2.threshold(img_u8, 50, 255, cv2.THRESH_BINARY)
#
#         # Project horizontally to find non-zero rows
#         row_sums = np.sum(thresh, axis=1)
#         non_zero_rows = np.where(row_sums > 0)[0]
#
#         if len(non_zero_rows) > 0:
#             y_top = non_zero_rows[0]
#             y_bottom = non_zero_rows[-1]
#             return img[y_top:y_bottom, :]
#         return img
#
#     def resize_and_pad_dynamic(self, img, target_size):
#         """Resizes and pads using image-specific background value."""
#         pad_value = self.get_background_value(img)
#         h, w = img.shape
#         scale = target_size / max(h, w)
#         new_h, new_w = int(h * scale), int(w * scale)
#         resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
#
#         final = np.full((target_size, target_size), pad_value, dtype=np.float32)
#         y_offset = (target_size - new_h) // 2
#         x_offset = (target_size - new_w) // 2
#         final[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
#         return final
#
#     def normalization(self, img):
#         windowed = np.clip(img, WIN_MIN, WIN_MAX)
#         min_val = np.min(windowed)
#         max_val = np.max(windowed)
#
#         normalized = (windowed - min_val) / (max_val - min_val)
#         return normalized
#
#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         nifti_path = row['nifti_path']
#         height_label = float(row['height_cm'])
#
#         try:
#             # 1. Load NIfTI
#             nii = nib.load(nifti_path)
#             img_data = nii.get_fdata()
#             header = nii.header
#             spacing = header.get_zooms()[:2]
#
#             # Handle Dimensions
#             if img_data.ndim >= 3: img_data = np.max(img_data, axis=-1) if img_data.shape[
#                                                                                -1] != 1 else img_data.squeeze()
#             if img_data.ndim > 2: img_data = np.squeeze(img_data)
#
#             # 2. Orient Upright
#             img_data, rotated = self.standardize_orientation_robust(img_data)
#             if rotated: spacing = (spacing[1], spacing[0])
#
#             spacing_tensor = torch.tensor(spacing, dtype=torch.float32)
#
#             # 3. Trim Empty Space (Deterministic)
#             # Remove air above head/below feet so we focus on the patient
#             img_data = self.trim_empty_vertical_space(img_data)
#
#             # 4. Random Vertical Crop
#             # Randomly cuts the height to simulate partial scans
#             img_data = self.random_vertical_crop(img_data)
#
#             # 5. Resize & Pad
#             img_data = self.resize_and_pad_dynamic(img_data, IMG_SIZE)
#
#             # 6. Normalize
#             img_data = self.normalization(img_data)
#
#             # 7. Albumentations (Flips, etc.)
#             img_data = img_data.astype(np.float32)[:, :, np.newaxis]
#             augmented = self.transform(image=img_data)
#             img_tensor = augmented['image']
#             img_tensor = img_tensor.repeat(3, 1, 1)
#
#             return img_tensor, spacing_tensor, torch.tensor(height_label, dtype=torch.float32)
#
#         except Exception as e:
#             print(f"Error loading {nifti_path}: {e}")
#             return torch.zeros((3, IMG_SIZE, IMG_SIZE)), torch.zeros(2), torch.tensor(0.0)
