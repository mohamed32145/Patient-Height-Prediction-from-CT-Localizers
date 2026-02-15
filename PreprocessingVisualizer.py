import cv2
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# ============================================================================
# CONFIGURATION - EDIT YOUR FILE PATH HERE
# ============================================================================
NIFTI_FILE = r"C:\Users\Lab2\Desktop\mohamed sliman\rambam_nifti_localizers\C25\28.10.2018\0000647F\C25_28.10.2018_0000647F.nii.gz"

PATIENT_ID = "C07"
SAVE_OUTPUT = False  # Set to True to save images
OUTPUT_DIR = r"C:\Users\Lab2\Desktop\preprocessing_output"

# Preprocessing parameters
IMG_SIZE = 256
TARGET_CROP_MM = 250  # FIXED PHYSICAL WIDTH (mm) instead of pixels 140


# ============================================================================
# IMPROVED PREPROCESSING FUNCTIONS
# ============================================================================

def load_nifti(path):
    """Load NIfTI file and extract 2D image"""
    nii = nib.load(path)
    img_data = nii.get_fdata()
    header = nii.header

    # Handle 3D volumes
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

    spacing = header.get_zooms()[:2]
    return img_data, spacing


def standardize_orientation_robust(img):
    """
    Robustly detects if body is horizontal, regardless of HU range.
    """
    # Normalize temp image to 0-255 just for detection
    img_u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Threshold at 20% intensity (works for both Raw and 8-bit)
    _, thresh = cv2.threshold(img_u8, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return img, False

    largest = max(contours, key=cv2.contourArea)
    _, _, w, h = cv2.boundingRect(largest)

    if w > h:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), True
    return img, False


def crop_to_spine_physical(img, spacing_x, target_width_mm=140):
    """
    Crops 140mm of anatomy, ensuring consistent scale.
    """
    if spacing_x <= 0: spacing_x = 1.0  # Safety fallback

    crop_w_pixels = int(target_width_mm / spacing_x)

    # Detection
    img_u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, thresh = cv2.threshold(img_u8, 50, 255, cv2.THRESH_BINARY)

    M = cv2.moments(thresh)
    if M["m00"] == 0:
        center_x = img.shape[1] // 2
    else:
        center_x = int(M["m10"] / M["m00"])

    row_sums = np.sum(thresh, axis=1)
    non_zero_rows = np.where(row_sums > 0)[0]
    if len(non_zero_rows) > 0:
        y_top, y_bottom = non_zero_rows[0], non_zero_rows[-1]
    else:
        y_top, y_bottom = 0, img.shape[0]

    h, w = img.shape
    x_start = max(0, center_x - crop_w_pixels // 2)
    x_end = min(w, center_x + crop_w_pixels // 2)

    return img[y_top:y_bottom, x_start:x_end]


def get_background_value(img):
    """
    Detects if the background is Air (-1024) or Black (0).
    """
    min_val = np.min(img)
    # If image has negative values like -1000, it's Air. Otherwise, it's 0.
    if min_val < -900:
        return -1024
    else:
        return min_val


def resize_and_pad_dynamic(img, target_size):
    """
    Resizes and pads using the IMAGE-SPECIFIC background value.
    """
    pad_value = get_background_value(img)

    h, w = img.shape
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Fill canvas with the detected background (Air or Black)
    final = np.full((target_size, target_size), pad_value, dtype=np.float32)

    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    final[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return final


def apply_adaptive_normalization(img):
    """
    Automatically handles Raw CT (12-bit) vs Processed (8-bit).
    """
    min_val = np.min(img)
    max_val = np.max(img)

    # CASE A: 8-bit Image (Max ~255) -> Just Scale
    if max_val <= 300 and min_val >= 0:
        if max_val == 0: return img
        return img / 255.0

    # CASE B: Raw CT (Max > 1000) -> Apply Bone Window
    else:
        WIN_MIN = -500
        WIN_MAX = 1300
        windowed = np.clip(img, WIN_MIN, WIN_MAX)
        normalized = (windowed - WIN_MIN) / (WIN_MAX - WIN_MIN)
        return normalized


# ============================================================================
# MAIN PROCESSING
# ============================================================================

print("=" * 70)
print("CT LOCALIZER PREPROCESSING VIEWER (UNIVERSAL FIX)")
print("=" * 70)
print(f"\nPatient: {PATIENT_ID}")
print(f"File: {NIFTI_FILE}")
print("\nProcessing...")

# Step 1: Load
print("  1. Loading NIfTI file...")
raw_img, spacing = load_nifti(NIFTI_FILE)
spacing_x = spacing[0]

# Step 2: Orientation (Robust)
print("  2. Checking orientation (Robust)...")
oriented, was_rotated = standardize_orientation_robust(raw_img)

# Step 3: Physical Crop
print(f"  3. Cropping to {TARGET_CROP_MM}mm spine ({spacing_x:.2f} mm/px)...")
cropped = crop_to_spine_physical(oriented, spacing_x, target_width_mm=TARGET_CROP_MM)

# Step 4: Resize & Pad (Dynamic)
print("  4. Resizing and padding (Dynamic Background)...")
resized = resize_and_pad_dynamic(cropped, IMG_SIZE)

# Step 5: Adaptive Normalization
print("  5. Applying Adaptive Normalization...")
final = apply_adaptive_normalization(resized)

print("\nProcessing complete!")
print("=" * 70)

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\nGenerating visualizations...\n")

# Figure 1: All Steps
fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 4, figure=fig, hspace=0.25, wspace=0.25)

# Plot each step
steps = [
    (raw_img, '1. Raw NIfTI', f'Shape: {raw_img.shape}\nRange: [{raw_img.min():.0f}, {raw_img.max():.0f}] HU', None),
    (oriented, '2. Orientation', f'Rotated: {"YES" if was_rotated else "NO"}\nShape: {oriented.shape}', None),
    (cropped, '3. Physical Crop', f'Shape: {cropped.shape}\nTarget: {TARGET_CROP_MM}mm', None),
    (resized, '4. Resized & Padded', f'Shape: {resized.shape}\nDynamic Pad Val', None),
    (final, '5. Final Output', f'Shape: {final.shape}\nReady for ResNet50', (0, 1)),
]

for idx, (img, title, info, vrange) in enumerate(steps):
    ax = fig.add_subplot(gs[idx // 4, idx % 4])

    if vrange:
        ax.imshow(img, cmap='gray', vmin=vrange[0], vmax=vrange[1])
    else:
        # Auto-scale display for raw steps
        ax.imshow(img, cmap='gray')

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')

    ax.text(0.02, 0.98, info, transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add summary
ax_summary = fig.add_subplot(gs[1, 1:3])
ax_summary.axis('off')
summary_text = f"""
PREPROCESSING SUMMARY - Patient {PATIENT_ID}

Original Image:
  • Shape: {raw_img.shape}
  • HU Range: [{raw_img.min():.1f}, {raw_img.max():.1f}]
  • Pixel Spacing: ({spacing[0]:.2f}, {spacing[1]:.2f})

Transformations Applied:
  ✓ Orientation: {"Rotated 90°" if was_rotated else "No rotation needed"}
  ✓ Physical Crop: {TARGET_CROP_MM} mm (Consistent scale)
  ✓ Padding: Dynamic (Auto-detected background)
  ✓ Normalization: Adaptive (Handles Raw vs 8-bit)
  ✓ Resize: {IMG_SIZE}×{IMG_SIZE}

Final Output:
  • Shape: (3, {IMG_SIZE}, {IMG_SIZE})
  • Value Range: [{final.min():.4f}, {final.max():.4f}]
  • Ready for: ResNet50 backbone
"""
ax_summary.text(0.1, 0.9, summary_text, transform=ax_summary.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

fig.suptitle(f'Universal Preprocessing Pipeline - Patient {PATIENT_ID}',
             fontsize=14, fontweight='bold')

if SAVE_OUTPUT:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f'{PATIENT_ID}_all_steps.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")

# Figure 2: Before/After Comparison
fig2, axes = plt.subplots(1, 2, figsize=(14, 7))

axes[0].imshow(raw_img, cmap='gray')
axes[0].set_title('Before: Raw NIfTI Image', fontsize=12, fontweight='bold')
axes[0].axis('off')
axes[0].text(0.02, 0.98,
             f'Shape: {raw_img.shape}\nHU: [{raw_img.min():.0f}, {raw_img.max():.0f}]',
             transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

axes[1].imshow(final, cmap='gray', vmin=0, vmax=1)
axes[1].set_title('After: Preprocessed Image', fontsize=12, fontweight='bold')
axes[1].axis('off')
axes[1].text(0.02, 0.98,
             f'Shape: {final.shape}\nValues: [0, 1]\nRotated: {"Yes" if was_rotated else "No"}',
             transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

fig2.suptitle(f'Before/After Comparison - Patient {PATIENT_ID}',
              fontsize=14, fontweight='bold')
plt.tight_layout()

if SAVE_OUTPUT:
    save_path = os.path.join(OUTPUT_DIR, f'{PATIENT_ID}_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE!")
print("=" * 70)
print(f"\nPatient: {PATIENT_ID}")
print(f"Raw image: {raw_img.shape}, HU range: [{raw_img.min():.1f}, {raw_img.max():.1f}]")
print(f"Final image: {final.shape}, normalized: [{final.min():.4f}, {final.max():.4f}]")
print(f"Rotated: {'YES (90° clockwise)' if was_rotated else 'NO (already vertical)'}")
print(f"Pixel spacing: ({spacing[0]:.2f}, {spacing[1]:.2f})")

if SAVE_OUTPUT:
    print(f"\n✓ Images saved to: {OUTPUT_DIR}")
else:
    print("\nTip: Set SAVE_OUTPUT = True to save images to disk")

print("=" * 70)

plt.show()