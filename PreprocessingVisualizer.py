import os
import json
import random

import cv2
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Patch

# ============================================================================
# CONFIGURATION
# ============================================================================
NIFTI_FILE = r"C:\Users\Lab2\Desktop\mohamed sliman\rambam_nifti_localizers\C18\07.03.2017\00008408\C18_07.03.2017_00008408.nii.gz"

PATIENT_ID = "C18"
SAVE_OUTPUT = True
# Figures are written next to the paper so they can be dropped straight into LaTeX.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "paper", "figures")

SHOW_SUPTITLE = True  # Set False when exporting for the paper (caption covers it)

# Pipeline parameters matching the PyTorch dataset config (config.py)
IMG_SIZE = 384
WIN_MIN = -500          # 400 - 1800 // 2
WIN_MAX = 1300          # 400 + 1800 // 2
TARGET_PIXEL_SPACING_MM = 1.0
DEMO_CROP_RATIO = 0.75  # Fixed, centered crop so the augmentation step is visible

# Operation categories -> colors (shown as a legend so the figure is self-explanatory)
CAT_COLORS = {
    "input":     "#7f8c8d",  # grey   - data loading
    "intensity": "#e08e3c",  # orange - intensity transform
    "geometric": "#3a6ea5",  # blue   - deterministic geometric transform
    "augment":   "#3a9e5c",  # green  - stochastic augmentation (train only)
}
CAT_LABELS = {
    "input":     "Load",
    "intensity": "Intensity transform",
    "geometric": "Geometric (deterministic)",
    "augment":   "Augmentation (train only)",
}


# ============================================================================
# PREPROCESSING FUNCTIONS (Mirrored from LocalizerDataset in dataset.py)
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
    """Find and load the DICOM JSON sidecar produced by dcm2niix, if present."""
    json_path = nifti_path.replace('.nii.gz', '.json').replace('.nii', '.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    return {}


def apply_windowing(img):
    """Clip to the bone Hounsfield-Unit window and rescale to [0, 1]."""
    img = np.clip(img, WIN_MIN, WIN_MAX)
    return (img - WIN_MIN) / (WIN_MAX - WIN_MIN)


def orient_from_metadata(img_2d, metadata):
    """
    Standardize to a head-up orientation using DICOM tags.
    Mirrors LocalizerDataset.orient_from_metadata.
    """
    if not metadata:
        return img_2d, False

    iop = metadata.get("ImageOrientationPatientDICOM", [1, 0, 0, 0, 1, 0])
    position = metadata.get("PatientPosition", "HFS")

    # Column vector (how the image is drawn top-to-bottom)
    c_x, c_y, c_z = iop[3], iop[4], iop[5]

    rotated = False
    # If X or Y dominates Z, the spine is drawn sideways -> rotate upright.
    if abs(c_x) > abs(c_z) or abs(c_y) > abs(c_z):
        img_2d = cv2.rotate(img_2d, cv2.ROTATE_90_CLOCKWISE)
        rotated = True

    if position.startswith("FFS"):
        img_2d = cv2.flip(img_2d, 0)  # Feet-first -> flip vertically

    return img_2d, rotated


def random_vertical_crop(img, demo_mode=True):
    """
    Randomly crop the height to simulate partial scans (train-time augmentation).
    demo_mode forces a fixed, centered crop so the step is clearly visible.
    """
    h, w = img.shape
    if demo_mode:
        new_h = int(h * DEMO_CROP_RATIO)
        y_start = (h - new_h) // 2
    else:
        crop_ratio = random.uniform(0.6, 1.0)
        new_h = int(h * crop_ratio)
        y_start = random.randint(0, h - new_h)
    return img[y_start:y_start + new_h, :]


def resample_to_target_spacing(img, spacing, target_spacing_mm=TARGET_PIXEL_SPACING_MM):
    """Resample to isotropic target pixel spacing (physically consistent scale)."""
    sx, sy = float(spacing[0]), float(spacing[1])
    h, w = img.shape
    new_w = max(1, int(round(w * (sx / target_spacing_mm))))
    new_h = max(1, int(round(h * (sy / target_spacing_mm))))
    resampled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resampled, (target_spacing_mm, target_spacing_mm)


def get_background_value(img):
    """Detect whether the background is air (-1024) or already black (0)."""
    min_val = np.min(img)
    return -1024 if min_val < -900 else min_val


def resize_pad_dynamic_with_spacing(img, spacing, target_size):
    """Aspect-preserving resize + symmetric padding, recomputing effective spacing."""
    pad_value = get_background_value(img)
    h, w = img.shape
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    final = np.full((target_size, target_size), pad_value, dtype=np.float32)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    final[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    new_spacing = (spacing[0] / scale, spacing[1] / scale)
    return final, new_spacing


# ============================================================================
# FIGURE
# ============================================================================

def build_steps():
    """Run the pipeline once and return the per-step records used for plotting."""
    raw_img, spacing = load_nifti(NIFTI_FILE)
    json_meta = load_json_metadata(NIFTI_FILE)

    windowed_img = apply_windowing(raw_img)

    oriented_img, was_rotated = orient_from_metadata(windowed_img, json_meta)
    if was_rotated:
        spacing = (spacing[1], spacing[0])

    cropped_img = random_vertical_crop(oriented_img, demo_mode=True)

    resampled_img, spacing_iso = resample_to_target_spacing(cropped_img, spacing)

    final_img, new_spacing = resize_pad_dynamic_with_spacing(resampled_img, spacing_iso, IMG_SIZE)

    # (image, title, category, metric-overlay, explanation-below, vrange)
    return [
        (raw_img, "1. Raw NIfTI", "input",
         f"{raw_img.shape}\n[{raw_img.min():.0f}, {raw_img.max():.0f}] HU",
         "Load the 2D localizer\nfrom NIfTI and read\npixel spacing from\nthe header.", None),

        (windowed_img, "2. Bone Windowing", "intensity",
         "[0.0, 1.0]",
         "Clip to bone window\n[-500, 1300] HU and\nrescale intensities\nto [0, 1].", (0, 1)),

        (oriented_img, "3. Orientation", "geometric",
         f"rotated: {'yes' if was_rotated else 'no'}",
         "Standardize to head-up\nusing DICOM tags; swap\nspacing axes if the\nimage was rotated.", (0, 1)),

        (cropped_img, "4. Vertical Crop", "augment",
         f"{cropped_img.shape}\n75% (demo)",
         "Random 60-100% height\ncrop (train only) to\nremove boundary cues\n(the 'Ruler Effect').", (0, 1)),

        (resampled_img, "5. Resample 1.0 mm", "geometric",
         f"{resampled_img.shape}\n1.0 mm iso",
         "Resample to 1.0 mm\nisotropic spacing for a\nphysically consistent\nscale across scans.", (0, 1)),

        (final_img, f"6. Resize & Pad {IMG_SIZE}", "geometric",
         f"{final_img.shape}\nspacing {new_spacing[0]:.2f} mm",
         f"Aspect-preserving resize\nto {IMG_SIZE}x{IMG_SIZE} with\nbackground padding;\nrecompute spacing.", (0, 1)),
    ]


def render(steps):
    n = len(steps)

    # ---- Geometry in figure-fraction coordinates (manual layout for clean arrows)
    left_m, right_m = 0.015, 0.015
    gap = 0.020
    panel_w = (1.0 - left_m - right_m - (n - 1) * gap) / n
    panel_bottom, panel_h = 0.36, 0.40
    panel_top = panel_bottom + panel_h
    arrow_y = panel_bottom + panel_h / 2.0

    fig = plt.figure(figsize=(20, 6.2))

    lefts = [left_m + i * (panel_w + gap) for i in range(n)]

    for i, (img, title, cat, metric, desc, vrange) in enumerate(steps):
        ax = fig.add_axes([lefts[i], panel_bottom, panel_w, panel_h])

        if vrange:
            ax.imshow(img, cmap="gray", vmin=vrange[0], vmax=vrange[1], aspect="auto")
        else:
            ax.imshow(img, cmap="gray", aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
            spine.set_linewidth(1.0)

        # Colored header band with the step name
        ax.set_title(
            title, fontsize=12, fontweight="bold", color="white", pad=7,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=CAT_COLORS[cat], edgecolor="none"),
        )

        # Small metric overlay (shape / range) inside the image
        ax.text(
            0.035, 0.965, metric, transform=ax.transAxes, va="top", ha="left",
            fontsize=8, color="black",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.78, edgecolor="none"),
        )

        # Brief explanation underneath each panel
        fig.text(
            lefts[i] + panel_w / 2.0, panel_bottom - 0.035, desc,
            ha="center", va="top", fontsize=9.5, color="#222222", linespacing=1.35,
        )

    # ---- Flow arrows between consecutive panels
    for i in range(n - 1):
        x0 = lefts[i] + panel_w
        x1 = lefts[i + 1]
        arrow = FancyArrowPatch(
            (x0 + 0.0015, arrow_y), (x1 - 0.0015, arrow_y),
            transform=fig.transFigure, arrowstyle="-|>", mutation_scale=24,
            lw=2.4, color="#333333", shrinkA=0, shrinkB=0,
        )
        fig.add_artist(arrow)

    # ---- Category legend
    handles = [Patch(facecolor=CAT_COLORS[k], edgecolor="none", label=CAT_LABELS[k])
               for k in ["input", "intensity", "geometric", "augment"]]
    fig.legend(
        handles=handles, loc="lower center", ncol=4, frameon=False,
        fontsize=11, bbox_to_anchor=(0.5, 0.005),
    )

    if SHOW_SUPTITLE:
        fig.suptitle(
            f"CT Localizer Preprocessing Pipeline  (Patient {PATIENT_ID})",
            fontsize=15, fontweight="bold", y=0.975,
        )

    return fig


if __name__ == "__main__":
    print("=" * 70)
    print("CT LOCALIZER PREPROCESSING VIEWER (DATASET MIRROR)")
    print("=" * 70)

    steps = build_steps()
    fig = render(steps)

    if SAVE_OUTPUT:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        png_path = os.path.join(OUTPUT_DIR, "preprocessing_pipeline.png")
        pdf_path = os.path.join(OUTPUT_DIR, "preprocessing_pipeline.pdf")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")  # vector version for LaTeX
        print(f"Saved: {png_path}")
        print(f"Saved: {pdf_path}")

    plt.show()
