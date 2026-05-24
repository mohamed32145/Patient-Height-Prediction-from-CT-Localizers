"""
MONAI dictionary-based dataset for CT localizer height prediction.

This is a refactor of dataset.py that keeps every domain-specific step
(JSON-driven orientation, windowing, random vertical crop, resample to
1 mm, dynamic pad with spacing tracking) but expresses each step as a
MONAI `MapTransform`. The pipeline composes through `monai.transforms.Compose`
so it can be combined with MONAI's built-in augmentations and inspected
with MONAI tooling.

Each sample is a dict with keys:
    image    -> torch.FloatTensor [3, H, W], ImageNet-normalized
    spacing  -> torch.FloatTensor [2]   (mm per pixel, post resize/pad)
    label    -> torch.FloatTensor scalar (height in cm)
"""

import json
import os
import random
from typing import Dict, Hashable, List, Mapping

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
import torch

from monai.config import KeysCollection
from monai.data import Dataset
from monai.transforms import (
    Compose,
    MapTransform,
    NormalizeIntensityd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
)

from config import (
    AUG_BRIGHTNESS_CONTRAST_PROB,
    AUG_HORIZONTAL_FLIP_PROB,
    AUG_ROTATE_LIMIT,
    AUG_SHIFT_LIMIT,
    AUG_SHIFT_SCALE_ROTATE_PROB,
    IMG_SIZE,
    TARGET_PIXEL_SPACING_MM,
    WIN_MAX,
    WIN_MIN,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_background_value(img: np.ndarray) -> float:
    """Detects if background is Air (-1024) or near-zero black."""
    min_val = float(np.min(img))
    return -1024.0 if min_val < -900 else min_val


def _load_json_sidecar(nifti_path: str) -> Dict:
    json_path = nifti_path.replace(".nii.gz", ".json").replace(".nii", ".json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Custom MapTransforms
# ---------------------------------------------------------------------------
class LoadLocalizerSliced(MapTransform):
    """
    Load a NIfTI from `nifti_path`, reduce to a single 2D slice if needed,
    and emit `image` (HxW float32) plus initial `spacing` (sx, sy in mm).

    Replaces LocalizerDataset.__getitem__ steps 1 in dataset.py.
    """

    def __init__(self, keys: KeysCollection = ("nifti_path",), allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, object]) -> Dict[Hashable, object]:
        d = dict(data)
        nifti_path = d["nifti_path"]
        nii = nib.load(nifti_path)
        img = nii.get_fdata()
        header = nii.header

        if img.ndim >= 3:
            img = img.squeeze() if img.shape[-1] == 1 else np.max(img, axis=-1)
        if img.ndim > 2:
            img = img[..., img.shape[-1] // 2]
        elif img.ndim < 2:
            img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

        spacing = header.get_zooms()[:2]

        d["image"] = img.astype(np.float32)
        d["spacing"] = (float(spacing[0]), float(spacing[1]))
        d["json_meta"] = _load_json_sidecar(nifti_path)
        return d


class WindowIntensityd(MapTransform):
    """Clip to [WIN_MIN, WIN_MAX] and rescale to [0, 1]."""

    def __init__(self, keys: KeysCollection = ("image",), win_min: float = WIN_MIN, win_max: float = WIN_MAX):
        super().__init__(keys)
        self.win_min = win_min
        self.win_max = win_max

    def __call__(self, data: Mapping[Hashable, object]) -> Dict[Hashable, object]:
        d = dict(data)
        for k in self.keys:
            img = np.clip(d[k], self.win_min, self.win_max)
            d[k] = (img - self.win_min) / (self.win_max - self.win_min)
        return d


class JSONOrientationd(MapTransform):
    """
    Apply rotation / flip derived from the DICOM tags in the JSON sidecar
    and swap the recorded spacing accordingly. Mirrors
    LocalizerDataset.orient_from_metadata.
    """

    def __init__(self, image_key: str = "image", meta_key: str = "json_meta", spacing_key: str = "spacing"):
        super().__init__(keys=(image_key,))
        self.image_key = image_key
        self.meta_key = meta_key
        self.spacing_key = spacing_key

    def __call__(self, data: Mapping[Hashable, object]) -> Dict[Hashable, object]:
        d = dict(data)
        img = d[self.image_key]
        meta = d.get(self.meta_key) or {}
        spacing = d[self.spacing_key]

        if not meta:
            return d

        iop = meta.get("ImageOrientationPatientDICOM", [1, 0, 0, 0, 1, 0])
        position = meta.get("PatientPosition", "HFS")
        c_x, c_y, c_z = iop[3], iop[4], iop[5]

        rotated = False
        if abs(c_x) > abs(c_z) or abs(c_y) > abs(c_z):
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            rotated = True

        if str(position).startswith("FFS"):
            img = cv2.flip(img, 0)

        d[self.image_key] = img
        if rotated:
            d[self.spacing_key] = (spacing[1], spacing[0])
        return d


class RandVerticalCropd(MapTransform):
    """Random vertical crop simulating partial-body scans (train only)."""

    def __init__(
        self,
        keys: KeysCollection = ("image",),
        min_fraction: float = 0.6,
        max_fraction: float = 1.0,
        prob: float = 1.0,
    ):
        super().__init__(keys)
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction
        self.prob = prob

    def __call__(self, data: Mapping[Hashable, object]) -> Dict[Hashable, object]:
        d = dict(data)
        if random.random() > self.prob:
            return d
        for k in self.keys:
            img = d[k]
            h, _ = img.shape
            crop_ratio = random.uniform(self.min_fraction, self.max_fraction)
            new_h = int(h * crop_ratio)
            y_start = random.randint(0, h - new_h)
            d[k] = img[y_start : y_start + new_h, :]
        return d


class ResamplePixelSpacingd(MapTransform):
    """Resample image to the target isotropic pixel spacing (mm)."""

    def __init__(
        self,
        image_key: str = "image",
        spacing_key: str = "spacing",
        target_spacing_mm: float = TARGET_PIXEL_SPACING_MM,
    ):
        super().__init__(keys=(image_key,))
        self.image_key = image_key
        self.spacing_key = spacing_key
        self.target = float(target_spacing_mm)

    def __call__(self, data: Mapping[Hashable, object]) -> Dict[Hashable, object]:
        d = dict(data)
        img = d[self.image_key]
        sx, sy = d[self.spacing_key]
        h, w = img.shape
        new_w = max(1, int(round(w * (sx / self.target))))
        new_h = max(1, int(round(h * (sy / self.target))))
        d[self.image_key] = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        d[self.spacing_key] = (self.target, self.target)
        return d


class ResizeWithPadAndUpdateSpacingd(MapTransform):
    """
    Aspect-preserving resize to `target_size` longest side, then pad to a
    square using a per-image background value. Recomputes the effective
    pixel spacing so the model's metadata branch receives accurate mm/px.
    """

    def __init__(
        self,
        image_key: str = "image",
        spacing_key: str = "spacing",
        target_size: int = IMG_SIZE,
    ):
        super().__init__(keys=(image_key,))
        self.image_key = image_key
        self.spacing_key = spacing_key
        self.target_size = int(target_size)

    def __call__(self, data: Mapping[Hashable, object]) -> Dict[Hashable, object]:
        d = dict(data)
        img = d[self.image_key]
        spacing = d[self.spacing_key]

        pad_value = _get_background_value(img)
        h, w = img.shape
        scale = self.target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((self.target_size, self.target_size), pad_value, dtype=np.float32)
        y_off = (self.target_size - new_h) // 2
        x_off = (self.target_size - new_w) // 2
        canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized

        d[self.image_key] = canvas
        d[self.spacing_key] = (spacing[0] / scale, spacing[1] / scale)
        return d


class ImageToCHW3d(MapTransform):
    """Take a single-channel HxW float image and produce a 3xHxW tensor."""

    def __init__(self, keys: KeysCollection = ("image",)):
        super().__init__(keys)

    def __call__(self, data: Mapping[Hashable, object]) -> Dict[Hashable, object]:
        d = dict(data)
        for k in self.keys:
            img = d[k]
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img).float()
            if img.ndim == 2:
                img = img.unsqueeze(0)
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            d[k] = img.contiguous()
        return d


class SpacingAndLabelToTensord(MapTransform):
    """Convert the spacing tuple and the scalar label to tensors."""

    def __init__(self, spacing_key: str = "spacing", label_key: str = "label"):
        super().__init__(keys=(spacing_key, label_key))
        self.spacing_key = spacing_key
        self.label_key = label_key

    def __call__(self, data: Mapping[Hashable, object]) -> Dict[Hashable, object]:
        d = dict(data)
        spacing = d[self.spacing_key]
        d[self.spacing_key] = torch.tensor(spacing, dtype=torch.float32)
        d[self.label_key] = torch.tensor(float(d[self.label_key]), dtype=torch.float32)
        return d


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------
def build_transforms(is_train: bool) -> Compose:
    """Return the dict-based pipeline for a train or eval split."""

    # 1) Common deterministic preprocessing — same for train and eval.
    common: List = [
        LoadLocalizerSliced(),
        WindowIntensityd(keys=("image",)),
        JSONOrientationd(),
    ]

    # 2) Random vertical crop is augmentation-only; runs on raw spacing
    #    just like the original pipeline.
    if is_train:
        common.append(RandVerticalCropd(keys=("image",), prob=1.0))

    # 3) Resample to 1 mm isotropic, then resize + dynamic pad to IMG_SIZE
    #    while tracking the updated pixel spacing for the model's metadata
    #    branch.
    common.extend(
        [
            ResamplePixelSpacingd(),
            ResizeWithPadAndUpdateSpacingd(),
            ImageToCHW3d(keys=("image",)),
        ]
    )

    # 4) Per-channel ImageNet normalization, applied to the 3xHxW tensor.
    common.append(
        NormalizeIntensityd(
            keys=("image",),
            subtrahend=IMAGENET_MEAN,
            divisor=IMAGENET_STD,
            channel_wise=True,
        )
    )

    # 5) Augmentations layered on top of the normalized tensor, mirroring
    #    the Albumentations stack in dataset.py.
    if is_train:
        common.extend(
            [
                RandFlipd(keys=("image",), prob=AUG_HORIZONTAL_FLIP_PROB, spatial_axis=1),
                RandAffined(
                    keys=("image",),
                    prob=AUG_SHIFT_SCALE_ROTATE_PROB,
                    rotate_range=(np.deg2rad(AUG_ROTATE_LIMIT),),
                    translate_range=(AUG_SHIFT_LIMIT * IMG_SIZE, AUG_SHIFT_LIMIT * IMG_SIZE),
                    padding_mode="zeros",
                ),
                RandScaleIntensityd(keys=("image",), factors=0.2, prob=AUG_BRIGHTNESS_CONTRAST_PROB),
                RandShiftIntensityd(keys=("image",), offsets=0.2, prob=AUG_BRIGHTNESS_CONTRAST_PROB),
                RandGaussianNoised(keys=("image",), prob=0.3, mean=0.0, std=0.005),
            ]
        )

    # 6) Final tensor conversion for spacing + label.
    common.append(SpacingAndLabelToTensord())

    return Compose(common)


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------
def df_to_data_list(df: pd.DataFrame) -> List[Dict]:
    """Convert one of the per-fold dataframes into MONAI's list-of-dicts format."""
    return [
        {"nifti_path": row["nifti_path"], "label": float(row["height_cm"])}
        for _, row in df.iterrows()
    ]


class LocalizerDatasetMONAI(Dataset):
    """Thin wrapper that turns a per-fold dataframe into a MONAI Dataset."""

    def __init__(self, df: pd.DataFrame, is_train: bool = False):
        data = df_to_data_list(df)
        transform = build_transforms(is_train=is_train)
        super().__init__(data=data, transform=transform)


if __name__ == "__main__":
    # Quick smoke test on a fake dataframe — useful for tinkering with the
    # transform stack without spinning up the full pipeline.
    print("MONAI dataset module loaded. Use LocalizerDatasetMONAI(df, is_train=...).")
