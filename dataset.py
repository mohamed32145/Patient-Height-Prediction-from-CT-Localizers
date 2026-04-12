import json
import os
import random

import albumentations as A
import cv2
import nibabel as nib
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from config import (
    AUGMENTATION_PROBABILITY,
    AUG_BRIGHTNESS_LIMIT,
    AUG_CONTRAST_LIMIT,
    AUG_ERASE_MAX_FRACTION,
    AUG_ROTATE_LIMIT,
    HU_MAX,
    HU_MIN,
    IMG_HEIGHT,
    IMG_WIDTH,
    TARGET_PIXEL_SPACING_MM,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class LocalizerDataset(Dataset):
    """Dataset for CT localizer preprocessing and loading."""

    def __init__(self, df, is_train=False):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train

        erase_holes = max(1, int(IMG_HEIGHT * AUG_ERASE_MAX_FRACTION / 32))
        erase_w = max(8, int(IMG_WIDTH * AUG_ERASE_MAX_FRACTION))
        erase_h = max(8, int(IMG_HEIGHT * AUG_ERASE_MAX_FRACTION))

        if self.is_train:
            self.transform = A.Compose([
                A.Rotate(
                    limit=(-AUG_ROTATE_LIMIT, AUG_ROTATE_LIMIT),
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                    p=AUGMENTATION_PROBABILITY,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=AUG_BRIGHTNESS_LIMIT,
                    contrast_limit=AUG_CONTRAST_LIMIT,
                    p=AUGMENTATION_PROBABILITY,
                ),
                A.CoarseDropout(
                    num_holes_range=(1, erase_holes),
                    hole_height_range=(8, erase_h),
                    hole_width_range=(8, erase_w),
                    fill=0,
                    p=0.25,
                ),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.df)

    @staticmethod
    def load_json_metadata(nifti_path):
        json_path = nifti_path.replace('.nii.gz', '.json').replace('.nii', '.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                return json.load(f)
        return {}

    @staticmethod
    def resample_to_target_spacing(img, spacing, target_spacing_mm=TARGET_PIXEL_SPACING_MM):
        sx, sy = float(spacing[0]), float(spacing[1])
        h, w = img.shape
        new_w = max(1, int(round(w * (sx / target_spacing_mm))))
        new_h = max(1, int(round(h * (sy / target_spacing_mm))))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def center_crop_or_pad(img, target_h=IMG_HEIGHT, target_w=IMG_WIDTH, pad_value=0):
        h, w = img.shape

        if h < target_h or w < target_w:
            out = np.full((max(h, target_h), max(w, target_w)), pad_value, dtype=img.dtype)
            y0 = (out.shape[0] - h) // 2
            x0 = (out.shape[1] - w) // 2
            out[y0:y0 + h, x0:x0 + w] = img
            img = out
            h, w = img.shape

        start_y = (h - target_h) // 2
        start_x = (w - target_w) // 2
        return img[start_y:start_y + target_h, start_x:start_x + target_w]

    @staticmethod
    def make_clahe_rgb(img_u8):
        clahe_32 = cv2.createCLAHE(clipLimit=32, tileGridSize=(2, 2))
        clahe_64 = cv2.createCLAHE(clipLimit=64, tileGridSize=(1, 1))

        ch1 = img_u8
        ch2 = cv2.add(img_u8, clahe_32.apply(img_u8))
        ch3 = cv2.add(img_u8, clahe_64.apply(img_u8))

        return np.stack([ch1, ch2, ch3], axis=-1)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        nifti_path = row['nifti_path']
        height_label = float(row['height_cm'])

        try:
            nii = nib.load(nifti_path)
            img = nii.get_fdata()
            spacing = nii.header.get_zooms()[:2]
            if img.ndim >= 3:
                img = img.squeeze() if img.shape[-1] == 1 else np.max(img, axis=-1)
            if img.ndim > 2:
                img = img[..., img.shape[-1] // 2]

            img = np.clip(img, HU_MIN, HU_MAX)
            img = (img - HU_MIN) / float(HU_MAX - HU_MIN)
            img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)

            img_u8 = self.resample_to_target_spacing(img_u8, spacing)
            img_u8 = self.center_crop_or_pad(img_u8, IMG_HEIGHT, IMG_WIDTH, pad_value=0)
            img_rgb = self.make_clahe_rgb(img_u8)

            augmented = self.transform(image=img_rgb)
            img_tensor = augmented['image']

            return img_tensor, torch.tensor(height_label, dtype=torch.float32)

        except Exception as e:
            print(f'Error loading {nifti_path}: {e}')
            return torch.zeros((3, IMG_HEIGHT, IMG_WIDTH)), torch.tensor(0.0)
