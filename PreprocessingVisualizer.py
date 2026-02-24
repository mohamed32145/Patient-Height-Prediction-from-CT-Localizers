import cv2
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import random

# ============================================================================
# CONFIGURATION
# ============================================================================
NIFTI_FILE = r"C:\Users\Lab2\Desktop\mohamed sliman\rambam_nifti_localizers\C18\07.03.2017\00008408\C18_07.03.2017_00008408.nii.gz"

PATIENT_ID = "C07"
SAVE_OUTPUT = False
OUTPUT_DIR = r"C:\Users\Lab2\Desktop\preprocessing_output"

# Pipeline parameters matching the PyTorch dataset config
IMG_SIZE = 256
WIN_MIN = -500
WIN_MAX = 1300
THRESHOLD_VALUE = 50


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


def apply_windowing(img):
    """Clips to Hounsfield Unit window and scales to [0, 1]."""
    img = np.clip(img, WIN_MIN, WIN_MAX)
    return (img - WIN_MIN) / (WIN_MAX - WIN_MIN)


def standardize_orientation(img_2d):
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


def trim_empty_vertical_space(img):
    """Deterministic trim: Removes empty air above head and below feet."""
    img_u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, thresh = cv2.threshold(img_u8, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

    row_sums = np.sum(thresh, axis=1)
    non_zero_rows = np.where(row_sums > 0)[0]

    if len(non_zero_rows) > 0:
        y_top = non_zero_rows[0]
        y_bottom = non_zero_rows[-1]
        return img[y_top:y_bottom, :]
    return img


def random_vertical_crop(img, demo_mode=True):
    """
    Randomly crops height to simulate partial scans.
    demo_mode forces a visible 75% crop for the viewer.
    """
    h, w = img.shape

    if demo_mode:
        crop_ratio = 0.75
        new_h = int(h * crop_ratio)
        y_start = (h - new_h) // 2  # Center it for the demo
    else:
        crop_ratio = random.uniform(0.6, 1.0)
        new_h = int(h * crop_ratio)
        max_y_start = h - new_h
        y_start = random.randint(0, max_y_start)

    y_end = y_start + new_h
    return img[y_start:y_end, :]


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

# 1. Load NIfTI
raw_img, spacing = load_nifti(NIFTI_FILE)

# 2. Windowing
windowed_img = apply_windowing(raw_img)

# 3. Orientation
oriented_img, was_rotated = standardize_orientation(windowed_img)
if was_rotated:
    spacing = (spacing[1], spacing[0])

# 4. Trim Empty Space
trimmed_img = trim_empty_vertical_space(oriented_img)

# 5. Vertical Crop (Simulated Augmentation)
cropped_img = random_vertical_crop(trimmed_img, demo_mode=True)

# 6. Resize & Pad (Capturing updated spacing)
final_img, new_spacing = resize_pad_dynamic_with_spacing(cropped_img, spacing, IMG_SIZE)

# ============================================================================
# VISUALIZATION
# ============================================================================

# Figure 1: All Steps (Grid updated to handle 6 specific steps)
fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)

steps = [
    (raw_img, '1. Raw NIfTI', f'Shape: {raw_img.shape}\nRange: [{raw_img.min():.0f}, {raw_img.max():.0f}] HU', None),
    (windowed_img, '2. Windowing [0,1]', f'Range: [{windowed_img.min():.1f}, {windowed_img.max():.1f}]', (0, 1)),
    (oriented_img, '3. Orientation', f'Rotated: {"YES" if was_rotated else "NO"}\nShape: {oriented_img.shape}', (0, 1)),
    (trimmed_img, '4. Trim Air', f'Shape: {trimmed_img.shape}', (0, 1)),
    (cropped_img, '5. Vertical Crop (Aug)', f'Shape: {cropped_img.shape}\nDemo: 75% Ratio', (0, 1)),
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
summary_text = f"""
PIPELINE SUMMARY
Patient ID: {PATIENT_ID}

1. Original Data:
 • Size: {raw_img.shape}
 • Pixel Spacing: ({spacing[0]:.2f}, {spacing[1]:.2f})

2. Geometric Ops:
 • Rotation: {"90° CW" if was_rotated else "None"}
 • Trimmed Dead Space
 • Simulated Crop: 75% height

3. Final Tensor Prep:
 • Shape: ({IMG_SIZE}, {IMG_SIZE})
 • Values: [{final_img.min():.2f}, {final_img.max():.2f}]
 • New Spacing: ({new_spacing[0]:.4f}, {new_spacing[1]:.4f})
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