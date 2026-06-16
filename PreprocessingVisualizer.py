import cv2
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import random
import json

# ============================================================================
# CONFIGURATION
# ============================================================================
NIFTI_FILE = r"C:\Users\Lab2\Desktop\mohamed sliman\rambam_nifti_localizers\C5\15.06.2014\0000C1E8\C5_15.06.2014_0000C1E8.nii.gz"

PATIENT_ID = "C07"
SAVE_OUTPUT = False
OUTPUT_DIR = r"C:\Users\Lab2\Desktop\preprocessing_output"

# Pipeline parameters matching the PyTorch dataset config
IMG_SIZE = 384
TARGET_PIXEL_SPACING_MM = 1.0

# Explicit Bone Window (Level: 400, Width: 1800)
WIN_MIN = 400 - (1800 // 2)  # -500
WIN_MAX = 400 + (1800 // 2)  # 1300


# ============================================================================
# PREPROCESSING FUNCTIONS (Mirrored from LocalizerDataset)
# ============================================================================

def load_nifti(path):
    """Load NIfTI file, handle dimensions, and extract 2D image & spacing."""
    nii = nib.load(path)
    img_data = nii.get_fdata()
    header = nii.header

    if img_data.ndim >= 3:
        img_data = img_data.squeeze() if img_data.shape[-1] == 1 else np.max(img_data, axis=-1)

    if img_data.ndim > 2:
        img_data = img_data[..., img_data.shape[-1] // 2]
    elif img_data.ndim < 2:
        img_data = np.zeros((IMG_SIZE, IMG_SIZE))

    spacing = header.get_zooms()[:2]
    return img_data, spacing


def load_json_metadata(nifti_path):
    """Finds and loads the corresponding JSON sidecar file."""
    json_path = nifti_path.replace('.nii.gz', '.json').replace('.nii', '.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    return {}  # Return empty dict if missing


def apply_windowing(img):
    """Clips to Hounsfield Unit window and scales to [0, 1]."""
    img = np.clip(img, WIN_MIN, WIN_MAX)
    img = (img - WIN_MIN) / (WIN_MAX - WIN_MIN)

    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    return img


def orient_from_metadata(img_2d, metadata):
    """Determines rotation and flipping based strictly on DICOM tags."""
    rotated = False

    if not metadata:
        return img_2d, False

    iop = metadata.get("ImageOrientationPatientDICOM", [1, 0, 0, 0, 1, 0])
    position = metadata.get("PatientPosition", "HFS")

    c_x, c_y, c_z = iop[3], iop[4], iop[5]

    if abs(c_x) > abs(c_z) or abs(c_y) > abs(c_z):
        img_2d = cv2.rotate(img_2d, cv2.ROTATE_90_CLOCKWISE)
        rotated = True

    if position.startswith("FFS"):
        img_2d = cv2.flip(img_2d, 0)

    return img_2d, rotated


def random_vertical_crop(img, demo_mode=True):
    """Randomly crops height to simulate partial scans."""
    h, w = img.shape

    if demo_mode:
        crop_ratio = 0.75
        new_h = int(h * crop_ratio)
        y_start = (h - new_h) // 2
    else:
        min_fraction = 0.6
        max_fraction = 1.0
        crop_ratio = random.uniform(min_fraction, max_fraction)
        new_h = int(h * crop_ratio)
        max_y_start = h - new_h
        y_start = random.randint(0, max_y_start)

    y_end = y_start + new_h
    return img[y_start:y_end, :]


def resample_to_target_spacing(img, spacing, target_spacing_mm=TARGET_PIXEL_SPACING_MM):
    """Resample image to isotropic target pixel spacing before network resizing."""
    sx, sy = float(spacing[0]), float(spacing[1])
    h, w = img.shape

    new_w = max(1, int(round(w * (sx / target_spacing_mm))))
    new_h = max(1, int(round(h * (sy / target_spacing_mm))))

    resampled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resampled, (target_spacing_mm, target_spacing_mm)


def get_background_value(img):
    """Detects if background is Air or Black."""
    min_val = np.min(img)
    return -1024 if min_val < -900 else min_val


def resize_pad_dynamic_with_spacing(img, spacing, target_size):
    """Resizes, pads dynamically, and calculates updated spacing mapping."""
    pad_value = get_background_value(img)
    h, w = img.shape
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    final = np.full((target_size, target_size), pad_value, dtype=np.float32)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    final[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    new_spacing_x = spacing[0] / scale
    new_spacing_y = spacing[1] / scale

    return final, (new_spacing_x, new_spacing_y)


# ============================================================================
# MAIN PROCESSING
# ============================================================================

print("=" * 70)
print("CT LOCALIZER PREPROCESSING VIEWER (DATASET MIRROR)")
print("=" * 70)

# 1. Load NIfTI & Metadata
raw_img, original_spacing = load_nifti(NIFTI_FILE)
json_meta = load_json_metadata(NIFTI_FILE)

# 2. Windowing
windowed_img = apply_windowing(raw_img)

# 3. Orientation (Metadata Driven)
oriented_img, was_rotated = orient_from_metadata(windowed_img, json_meta)
spacing = (original_spacing[1], original_spacing[0]) if was_rotated else original_spacing

# 4. Vertical Crop (Simulated Augmentation)
cropped_img = random_vertical_crop(oriented_img, demo_mode=True)

# 5. Resample to Isotropic Target Spacing (1 mm)
resampled_img, iso_spacing = resample_to_target_spacing(cropped_img, spacing, TARGET_PIXEL_SPACING_MM)

# 6. Final Resize & Pad
final_img, final_spacing = resize_pad_dynamic_with_spacing(resampled_img, iso_spacing, IMG_SIZE)

# ============================================================================
# VISUALIZATION
# ============================================================================

fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)

steps = [
    (raw_img, '1. Raw NIfTI', f'Shape: {raw_img.shape}\nRange: [{raw_img.min():.0f}, {raw_img.max():.0f}]', None),
    (windowed_img, '2. Windowing [0,1]', f'Range: [{windowed_img.min():.1f}, {windowed_img.max():.1f}]', (0, 1)),
    (oriented_img, '3. Metadata Orient', f'Rotated: {"YES" if was_rotated else "NO"}\nShape: {oriented_img.shape}',
     (0, 1)),
    (cropped_img, '4. Vertical Crop', f'Shape: {cropped_img.shape}\nDemo: 75% Ratio', (0, 1)),
    (resampled_img, f'5. Resample ({TARGET_PIXEL_SPACING_MM}mm)', f'Shape: {resampled_img.shape}\nSpacing: (1.0, 1.0)',
     (0, 1)),
    (final_img, '6. Resize & Pad', f'Shape: {final_img.shape}\nFinal', (0, 1)),
]

for idx, (img, title, info, vrange) in enumerate(steps):
    # Lay out the 6 plots in the first 3 columns of the 2x4 grid
    row = idx // 3
    col = idx % 3
    ax = fig.add_subplot(gs[row, col])

    if vrange:
        ax.imshow(img, cmap='gray', vmin=vrange[0], vmax=vrange[1])
    else:
        ax.imshow(img, cmap='gray')

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')
    ax.text(0.02, 0.98, info, transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add summary in the remaining column of the grid
ax_summary = fig.add_subplot(gs[:, 3])
ax_summary.axis('off')

# Get patient position safely for text output
patient_pos = json_meta.get('PatientPosition', 'Missing/Unknown') if json_meta else 'No JSON Found'

summary_text = f"""
PIPELINE SUMMARY
Patient ID: {PATIENT_ID}

1. Original Data:
 • Size: {raw_img.shape}
 • Pixel Spacing: ({original_spacing[0]:.2f}, {original_spacing[1]:.2f})
 • Position: {patient_pos}

2. Geometric Ops:
 • Rotation: {"90° CW" if was_rotated else "None"}
 • Simulated Crop: 75% height
 • Resampling: {TARGET_PIXEL_SPACING_MM}mm isotropic

3. Final Tensor Prep:
 • Target Window: [-500, 1300] HU
 • Target Output Size: ({IMG_SIZE}, {IMG_SIZE})
 • Output Values: [{final_img.min():.2f}, {final_img.max():.2f}]
 • Final Eq. Spacing: ({final_spacing[0]:.4f}, {final_spacing[1]:.4f})
"""
ax_summary.text(0.0, 0.8, summary_text, transform=ax_summary.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

fig.suptitle(f'Dataset Preprocessing Pipeline - Patient {PATIENT_ID}', fontsize=14, fontweight='bold')

if SAVE_OUTPUT:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f'{PATIENT_ID}_all_steps.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')

plt.show()