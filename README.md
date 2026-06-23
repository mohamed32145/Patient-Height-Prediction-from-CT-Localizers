# Patient Height Prediction from CT Localizers

A deep learning system that estimates patient height in centimeters from **CT Localizer (Scout/Topogram)** images. This is a **regression** problem that fuses image features from 2D X-ray-like scans with physical pixel spacing metadata to produce scale-aware height predictions.

The project emphasizes **medical transfer learning**, **bias mitigation** against shortcut learning, and **model interpretability** via Grad-CAM.

## Project Goal

Predict patient height from a single CT scout image by learning **anatomical proportions** (spine length, femur size, pelvis width) rather than image artifacts or padding boundaries.

## Key Features

- **EfficientNetV2-S Backbone** with ImageNet pre-trained weights and configurable layer freezing for fine-tuning control
- **Vertical Ruler Architecture** -- a 1x1 convolution compresses 1280 channels to 256, followed by adaptive vertical pooling into 16 zones, producing a 4096-dim feature vector that encodes the body's vertical structure
- **Pixel Spacing Fusion** -- an MLP branch processes the physical `(x_spacing, y_spacing)` so the model understands real-world scale after resampling
- **Smart Preprocessing Pipeline:**
  - Bone Windowing (HU window: W=1800, L=400, range [-500, 1300])
  - DICOM metadata-driven orientation correction (handles HFS/FFS and horizontal scans)
  - Isotropic resampling to 1 mm/pixel before final resize to 384x384
  - Dynamic padding with background-aware fill values
- **Bias Mitigation** -- random vertical cropping (60-100% of body height) prevents the "Ruler Effect" where the model cheats by measuring padding borders
- **Grad-CAM Interpretability** to verify the model attends to spine, pelvis, and shoulders

## Architecture

```
CT Localizer (1ch) --> repeat to 3ch --> EfficientNetV2-S features (1280ch, 12x12)
                                              |
                                         1x1 Conv (256ch)
                                              |
                                      Vertical Pool (16x1)
                                              |
                                        Flatten (4096)
                                              |
Pixel Spacing (2) --> MLP (2 --> 32 --> 8) ---+--- Concat (4104) --> FC (512) --> Dropout --> FC (1)
                                                                                              |
                                                                                        Height (cm)
```

## Project Structure

```
.
|-- main.py                   # Entry point: runs the full cross-validation training pipeline
|-- config.py                 # All hyperparameters, paths, and training configuration
|-- model.py                  # HeightPredictor model (EfficientNetV2-S + Vertical Ruler + metadata fusion)
|-- dataset.py                # LocalizerDataset: NIfTI loading, windowing, orientation, resampling, augmentation
|-- Train.py                  # Training loop, evaluation, metrics (MAE, RMSE, median AE)
|-- utils.py                  # Data loading, patient-level stratified CV splits, result saving
|-- Inference.py              # HeightPredictor class for single/batch/ensemble inference
|-- Visualization.py          # Grad-CAM, dataset samples, predictions, error distributions, scatter plots
|-- Visualize.py              # CLI/interactive tool that ties all visualization modes together
|-- PreprocessingVisualizer.py # Step-by-step visualization of the preprocessing pipeline
|-- metadata.py               # Scans NIfTI/JSON files and exports scanner metadata to Excel
```

## Data Format

| Input | Format | Description |
|-------|--------|-------------|
| Images | NIfTI (`.nii` / `.nii.gz`) | 2D CT localizer scans with optional JSON sidecar for DICOM tags |
| Metadata | Excel (`.xlsx`) | Must contain `Patient_ID`, `Height` (cm), and `Localizer_Path_NIfTI` columns |

## Installation

```bash
# Clone the repository
git clone https://github.com/mohamed32145/Patient-Height-Prediction-from-CT-Localizers.git
cd Patient-Height-Prediction-from-CT-Localizers

# Install dependencies
pip install torch torchvision nibabel opencv-python albumentations pandas openpyxl numpy matplotlib
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

## Usage

### 1. Configure Paths

Edit [`config.py`](config.py) to set your data paths:

```python
EXCEL_PATH = Path('path/to/your/patient_list.xlsx')
NIFTI_ROOT = Path('path/to/your/nifti_localizers/')
```

### 2. Train

```bash
python main.py
```

This runs 4-fold patient-level cross-validation with stratified height-balanced splits. Each fold trains for 100 epochs with cosine annealing LR scheduling. Results are saved to `training_results_rotating.xlsx` and per-fold predictions to `experiments_height_pytorch/`.

### 3. Inference

```bash
python Inference.py --excel path/to/data.xlsx --model height_model_fold_1.pth --output predictions.xlsx
```

### 4. Visualization

```bash
# View preprocessed dataset samples
python Visualize.py --mode dataset --num-samples 5 --subset train

# Visualize Grad-CAM attention maps
python Visualize.py --mode gradcam --model height_model_fold_1.pth --num-samples 3

# Plot training curves from results
python Visualize.py --mode history --results training_results_rotating.xlsx

# View step-by-step preprocessing pipeline
python PreprocessingVisualizer.py
```

## Methodology

### Preprocessing Pipeline

1. **NIfTI Loading** -- extracts 2D slices from potentially 3D volumes (middle slice or MIP)
2. **Bone Windowing** -- clips to [-500, 1300] HU to isolate skeletal structures
3. **Orientation Correction** -- uses `ImageOrientationPatientDICOM` and `PatientPosition` from JSON sidecar to standardize vertical alignment
4. **Random Vertical Crop** (training only) -- simulates partial scans by cropping 60-100% of the body height
5. **Isotropic Resampling** -- resamples to 1 mm/pixel using the original NIfTI spacing, then resizes to 384x384 with updated spacing metadata

### Cross-Validation

Patient-level 4-fold CV with forced anchor patients per fold ensures:
- No data leakage between splits (all images from one patient stay in the same split)
- Height-stratified balancing across folds
- Reproducible test/validation anchors for consistent benchmarking

### Solving the "Ruler Effect"

Early models learned a shortcut: measuring the distance between the top and bottom padding borders to estimate height. Grad-CAM showed horizontal heatmap bars at the scan edges.

**Solution:** Random vertical cropping during training forces the model to see random sub-regions of the anatomy (e.g., just the torso). Since the crop boundaries no longer correlate with total height, the model must learn the *size* of anatomical structures (vertebrae, pelvis) to infer height. After applying this, Grad-CAM heatmaps shift to the **spine**, **pelvis**, and **shoulders**.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Backbone | EfficientNetV2-S (ImageNet pretrained) |
| Input Size | 384 x 384 |
| Optimizer | AdamW (lr=2e-4, weight_decay=1e-5) |
| Scheduler | Cosine Annealing (eta_min=3e-5) |
| Loss | MSE |
| Batch Size | 16 |
| Epochs | 100 |
| Dropout | 0.2 |
| Folds | 4 |

## Credits

- **[RadImageNet](https://github.com/BMEII-AI/RadImageNet)** -- pre-trained medical imaging backbone (explored in earlier experiments)
- **[TorchXRayVision](https://github.com/mlmed/torchxrayvision)** -- used in early experiments for comparison
- **[Albumentations](https://albumentations.ai/)** -- image augmentation pipeline

---

*Created by Mohamed Sliman*
