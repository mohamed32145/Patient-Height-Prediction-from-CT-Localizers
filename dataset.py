import cv2
import torch
import numpy as np
import pandas as pd
import nibabel as nib
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

import config  # Import your config file

def resolve_nifti_dir(localizer_dir_str: str) -> Path | None:
    """Slice after 'rambam_nifti_localizers' and join to NIFTI_ROOT."""
    if not isinstance(localizer_dir_str, str):
        return None
    # Handle both forward and backward slashes
    parts = localizer_dir_str.replace('\\', '/').split('rambam_nifti_localizers')
    if len(parts) < 2:
        return None

    tail = parts[-1].strip('/ ')
    candidate = config.NIFTI_ROOT / tail
    return candidate if candidate.exists() else None

def pick_nifti_file(nifti_dir: Path) -> Path | None:
    files = sorted(list(nifti_dir.glob('*.nii*')))
    return files[0] if files else None

def load_and_split_data():
    if not config.EXCEL_PATH.exists():
        raise FileNotFoundError(f"Excel file not found at {config.EXCEL_PATH}")
    if not config.NIFTI_ROOT.exists():
        raise FileNotFoundError(f"NIFTI root not found at {config.NIFTI_ROOT}")

    print("Loading Excel...")
    df = pd.read_excel(config.EXCEL_PATH, engine='openpyxl')
    df.columns = [str(c).strip() for c in df.columns]

    rows = []
    print("Resolving NIfTI paths...")
    for _, r in tqdm(df.iterrows(), total=len(df)):
        pid = str(r['Patient_ID'])
        height_cm = float(r['Height'])
        d = resolve_nifti_dir(r['Localizer_Path_NIfTI'])
        if d is None:
            continue
        f = pick_nifti_file(d)
        if f is None:
            continue
        rows.append({'Patient_ID': pid, 'nifti_path': str(f), 'height_cm': height_cm})

    data_df = pd.DataFrame(rows)
    print(f"Resolved NIfTI files for {len(data_df)} rows")

    # Splitting logic
    groups = data_df['Patient_ID'].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_val_idx, test_idx = next(gss.split(data_df, groups=groups))
    train_val_df = data_df.iloc[train_val_idx].reset_index(drop=True)
    test_df = data_df.iloc[test_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=42)
    train_idx, val_idx = next(gss2.split(train_val_df, groups=train_val_df['Patient_ID'].values))
    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df

class NiftiXRVDataset(Dataset):
    def __init__(self, df: pd.DataFrame, training: bool = False):
        self.df = df.reset_index(drop=True)
        self.training = training

        if self.training:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=5, p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([ToTensorV2()])

    def resize_pad(self, img):
        h, w = img.shape
        scale = config.IMG_SIZE / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        final_img = np.zeros((config.IMG_SIZE, config.IMG_SIZE), dtype=np.float32)
        y_offset = (config.IMG_SIZE - new_h) // 2
        x_offset = (config.IMG_SIZE - new_w) // 2
        final_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img
        return final_img

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = Path(row['nifti_path'])
        height_cm = float(row['height_cm'])

        nii = nib.load(str(path))
        data = nii.get_fdata()
        img2d = np.squeeze(data)
        if img2d.ndim > 2:
            img2d = img2d[..., img2d.shape[-1] // 2]

        img2d = np.clip(img2d, config.WIN_MIN, config.WIN_MAX)
        img2d = (img2d - config.WIN_MIN) / (config.WIN_MAX - config.WIN_MIN)
        img2d = self.resize_pad(img2d)

        augmented = self.transform(image=img2d)
        img_t = augmented['image']

        # XRV Scaling [-1024, 1024]
        img_t = (img_t * 2048) - 1024

        zooms = nii.header.get_zooms()
        if len(zooms) >= 2:
            spacing = np.array(zooms[:2], dtype=np.float32)
        else:
            spacing = np.array([1.0, 1.0], dtype=np.float32)
        spacing_t = torch.tensor(spacing, dtype=torch.float32)
        target = torch.tensor([height_cm], dtype=torch.float32)

        return img_t, spacing_t, target