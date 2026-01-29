
#  Patient Height Prediction from CT Localizers

This project implements a Deep Learning solution to estimate a patient's height based on **CT Localizer (Scout/Topogram)** images.

Unlike standard classification tasks, this is a **regression** problem that combines image data (2D X-Ray-like scans) with metadata (pixel spacing). The project explores the use of **RadImageNet** pre-trained weights versus standard ImageNet weights and places a strong emphasis on model interpretability using **Grad-CAM**.

##  Project Goal

To accurately predict patient height in centimeters from a single 2D CT scout image, ensuring the model learns **anatomical features** (spine length, femur size, body proportions) rather than relying on image artifacts or cropping boundaries.

##  Key Features

* **Hybrid Architecture:** Fuses ResNet50 image features (2048-dim) with Pixel Spacing metadata (8-dim) for scale-aware predictions.
* **Medical Transfer Learning:** Supports **RadImageNet** weights (pre-trained on 1.35M medical images) for better feature extraction compared to standard ImageNet.
* **Smart Preprocessing:**
* **Bone Windowing:** Applies specific Hounsfield Unit windows (W:1800, L:400) to highlight skeletal structures.
* **Contour-Based Cropping:** Automatically removes table artifacts and empty space to center the patient.


* **Bias Mitigation:** Implements robust augmentations (`RandomResizedCrop`) to prevent the **"Ruler Effect"** (where the model cheats by measuring the black padding borders).
* **Interpretability:** Integrated **Grad-CAM** visualization pipeline to verify the model is looking at the spine and pelvis.

##  Data Structure

The project expects data in the following format:

* **Images:** NIfTI (`.nii` / `.nii.gz`) format 2D localizers.
* **Metadata:** An Excel file containing `Patient_ID`, `Height` (target), and `Localizer_Path`.

##  Methodology

### 1. Preprocessing Pipeline

1. **NIfTI Loading:** Handles 3D volumes by extracting the middle slice or performing Maximum Intensity Projection (MIP).
2. **Windowing:** Clips intensities to `[-500, 1300]` HU to focus on bones.
3. **Smart Crop:** Uses OpenCV contours to detect the body and crop out the CT table and background air.

### 2. Model Architecture

We utilize a modified **ResNet50**:

* **Input:** 1-channel grayscale images are repeated to 3 channels to fit the ResNet architecture.
* **Backbone:** ResNet50 (RadImageNet or ImageNet weights).
* **Metadata Branch:** A small MLP processes the `(x_spacing, y_spacing)` tuple.
* **Fusion:** Backbone features and Metadata features are concatenated before the final regression head.

### 3. Solving the "Ruler Effect"

Initial experiments showed the model "cheating" by measuring the distance between top/bottom padding.

* **Solution:** We implemented **RandomResizedCrop** during training. This forces the model to see random "zoomed-in" sections of the anatomy (e.g., just the torso), preventing it from seeing the scan edges. This forces the network to learn the *size* of anatomical structures (vertebrae, pelvis) to infer height.



##  Results & Visualization

* **Shortcuts Detected:** Early models focused on top/bottom edges (horizontal heatmap bars).
* **Anatomy Learned:** After `RandomResizedCrop`, Grad-CAM heatmaps align with the **Spine**, **Pelvis**, and **Shoulders**, indicating true anatomical understanding.

##  Credits

* **RadImageNet:** For the pre-trained medical backbone. [GitHub](https://github.com/BMEII-AI/RadImageNet)
* **TorchXRayVision:** Used in early experiments for comparison. [GitHub](https://github.com/mlmed/torchxrayvision)
* **Albumentations:** For the powerful image augmentation pipeline.

---

*Created by Mohamed Sliman*
