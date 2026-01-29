import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import nibabel as nib
import torchxrayvision as xrv
import albumentations as A
import matplotlib.pyplot as plt

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# --- Configuration Constants ---
# Adjust these paths based on your local machine
BASE_DIR = Path(__file__).resolve().parent
EXCEL_PATH = BASE_DIR /'C:/Users/Lab2\Desktop/mohamed sliman/Patients_list_body_CT_localizers.xlsx'
NIFTI_ROOT = BASE_DIR / 'C:/Users/Lab2/Desktop/mohamed sliman/rambam_nifti_localizers'


EXPERIMENTS_DIR = BASE_DIR / 'experiments_height_pytorch'

# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0  # Set to 0 for debugging, increase for speed on Linux

# Bone Window Settings (W:1800, L:400) -> Range: [-500, 1300]
WIN_MIN = 400 - (1800 // 2)
WIN_MAX = 400 + (1800 // 2)


# --- 1. Data Preparation ---

def resolve_nifti_dir(localizer_dir_str: str) -> Path | None:
    """Slice after 'rambam_nifti_localizers' and join to NIFTI_ROOT."""
    if not isinstance(localizer_dir_str, str):
        return None
    # Handle both forward and backward slashes for cross-platform compatibility
    parts = localizer_dir_str.replace('\\', '/').split('rambam_nifti_localizers')
    if len(parts) < 2:
        return None

    tail = parts[-1].strip('/ ')
    candidate = NIFTI_ROOT / tail
    return candidate if candidate.exists() else None


def pick_nifti_file(nifti_dir: Path) -> Path | None:
    files = sorted(list(nifti_dir.glob('*.nii*')))
    return files[0] if files else None


def load_and_split_data():
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Excel file not found at {EXCEL_PATH}")
    if not NIFTI_ROOT.exists():
        raise FileNotFoundError(f"NIFTI root not found at {NIFTI_ROOT}")

    print("Loading Excel...")
    df = pd.read_excel(EXCEL_PATH, engine='openpyxl')
    df.columns = [str(c).strip() for c in df.columns]

    # Validation
    required_cols = ['Patient_ID', 'Height', 'Localizer_Path_NIfTI']
    for col in required_cols:
        assert col in df.columns, f"Missing '{col}' in Excel"

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

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=42)  # ~15% total for val
    train_idx, val_idx = next(gss2.split(train_val_df, groups=train_val_df['Patient_ID'].values))
    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


# --- 2. Dataset Class ---

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
            self.transform = A.Compose([
                ToTensorV2()
            ])

    def resize_pad(self, img):
        h, w = img.shape
        scale = IMG_SIZE / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        final_img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        y_offset = (IMG_SIZE - new_h) // 2
        x_offset = (IMG_SIZE - new_w) // 2
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

        img2d = np.clip(img2d, WIN_MIN, WIN_MAX)
        img2d = (img2d - WIN_MIN) / (WIN_MAX - WIN_MIN)
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


# --- 3. Model Definition ---

class XRVHeightRegressor(nn.Module):
    def __init__(self, spacing_dim=2):
        super().__init__()
        print("Loading TorchXRayVision model...")
        self.backbone = xrv.models.DenseNet(weights="densenet121-res224-all").to(DEVICE)
        self.backbone.eval()

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.embed_dim = 1024
        self.spacing_mlp = nn.Sequential(
            nn.BatchNorm1d(spacing_dim),
            nn.Linear(spacing_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim + 8, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, spacing):
        feats = self.backbone.features(x)
        feats = F.relu(feats, inplace=True)
        feats = F.adaptive_avg_pool2d(feats, (1, 1)).view(feats.size(0), -1)
        s = self.spacing_mlp(spacing)
        combined = torch.cat([feats, s], dim=1)
        out = self.head(combined)
        return out


# --- 4. Training & Viz Utilities ---

def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_mse, total_mae, total_n = 0.0, 0.0, 0

    for img_t, spacing_t, y_t in loader:
        img_t, spacing_t, y_t = img_t.to(DEVICE), spacing_t.to(DEVICE), y_t.to(DEVICE)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        pred = model(img_t, spacing_t)
        loss = F.mse_loss(pred, y_t)
        if is_train:
            loss.backward()
            optimizer.step()

        bs = img_t.size(0)
        total_mse += loss.item() * bs
        total_mae += (pred - y_t).abs().sum().item()
        total_n += bs

    return total_mse / total_n, total_mae / total_n


def compute_gradcam(model, input_image, spacing, target_layer):
    model.eval()
    activations_list, gradients_list = [], []

    def forward_hook(module, input, output):
        activations_list.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients_list.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    input_image.requires_grad_(True)
    output = model(input_image, spacing)
    model.zero_grad()
    output.backward(torch.ones_like(output), retain_graph=True)

    activations_np = activations_list[0].cpu().data.numpy()
    gradients_np = gradients_list[0].cpu().data.numpy()

    forward_handle.remove()
    backward_handle.remove()

    weights = np.mean(gradients_np, axis=(2, 3))[0]
    grad_cam = activations_np[0] * weights[:, None, None]
    grad_cam = np.maximum(grad_cam, 0)
    grad_cam = np.sum(grad_cam, axis=0)
    grad_cam = grad_cam - np.min(grad_cam)
    if np.max(grad_cam) != 0:
        grad_cam = grad_cam / np.max(grad_cam)
    grad_cam = cv2.resize(grad_cam, (IMG_SIZE, IMG_SIZE))
    return grad_cam


def visualize_sample(model, dataset, df):
    idx_to_visualize = 0
    sample_img_t, sample_spacing_t, sample_target_t = dataset[idx_to_visualize]
    sample_img_t = sample_img_t.unsqueeze(0).to(DEVICE)
    sample_spacing_t = sample_spacing_t.unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        pred_height = model(sample_img_t, sample_spacing_t).item()

    target_layer = model.backbone.features.norm5
    heatmap = compute_gradcam(model, sample_img_t, sample_spacing_t, target_layer)
    original_image_np = (sample_img_t.squeeze().cpu().detach().numpy() + 1024) / 2048

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image_np, cmap='gray')
    plt.title(f'True: {sample_target_t.item():.2f}cm, Pred: {pred_height:.2f}cm')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(original_image_np, cmap='gray')
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title('Grad-CAM')
    plt.axis('off')
    plt.show()

    print(f"Viz Patient: {df.iloc[idx_to_visualize]['Patient_ID']}")


# --- 5. Main Execution ---

def main():
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    train_df, val_df, test_df = load_and_split_data()

    train_dataset = NiftiXRVDataset(train_df, training=True)
    val_dataset = NiftiXRVDataset(val_df, training=False)
    test_dataset = NiftiXRVDataset(test_df, training=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=True)

    # 2. Setup Model
    model = XRVHeightRegressor(spacing_dim=2).to(DEVICE)

    # 3. Stage 1: Train Head Only
    print("\n--- Stage 1: Training Head Only ---")
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and 'backbone' not in n]
    optimizer = torch.optim.Adam(head_params, lr=1e-3)

    best_val_mae, patience, bad = float('inf'), 8, 0
    save_path_head = EXPERIMENTS_DIR / "best_xrv_head.pt"

    for epoch in range(1, 21):  # Shortened for demo, increase to 40
        tr_mse, tr_mae = run_epoch(model, train_loader, optimizer)
        va_mse, va_mae = run_epoch(model, val_loader, optimizer=None)
        print(f"[E{epoch:02d}] Train MAE {tr_mae:.2f} || Val MAE {va_mae:.2f}")

        if va_mae < best_val_mae - 1e-6:
            best_val_mae, bad = va_mae, 0
            torch.save(model.state_dict(), save_path_head)
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    # 4. Stage 2: Fine Tuning
    print("\n--- Stage 2: Fine Tuning Backbone ---")
    model.load_state_dict(torch.load(save_path_head, map_location=DEVICE))

    # Unfreeze specific layers
    for name, p in model.backbone.named_parameters():
        p.requires_grad = ('denseblock4' in name) or ('norm5' in name)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    best_val_mae_ft, bad = best_val_mae, 0
    save_path_ft = EXPERIMENTS_DIR / "best_xrv_finetuned.pt"

    for epoch in range(1, 6):
        tr_mse, tr_mae = run_epoch(model, train_loader, optimizer)
        va_mse, va_mae = run_epoch(model, val_loader, optimizer=None)
        print(f"[FT{epoch:02d}] Train MAE {tr_mae:.2f} || Val MAE {va_mae:.2f}")
        if va_mae < best_val_mae_ft - 1e-6:
            best_val_mae_ft, bad = va_mae, 0
            torch.save(model.state_dict(), save_path_ft)
        else:
            bad += 1
            if bad >= patience:
                break

    # 5. Final Evaluation & Visualization
    print("\n--- Final Evaluation ---")
    model.load_state_dict(torch.load(save_path_ft, map_location=DEVICE))
    model.eval()

    test_mse, test_mae = run_epoch(model, test_loader, optimizer=None)
    print(f"Final Test MAE: {test_mae:.2f} cm")

    # Run Visualization
    try:
        visualize_sample(model, test_dataset, test_df)
    except Exception as e:
        print(f"Visualization skipped: {e}")


if __name__ == "__main__":
    main()