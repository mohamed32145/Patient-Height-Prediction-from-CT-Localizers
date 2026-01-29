import torch
from pathlib import Path

# --- Paths ---
# Adjust these paths based on your local machine
BASE_DIR = Path(__file__).resolve().parent
EXCEL_PATH = BASE_DIR / 'C:/Users/Lab2/Desktop/mohamed sliman/Patients_list_body_CT_localizers.xlsx'
NIFTI_ROOT = BASE_DIR / 'C:/Users/Lab2/Desktop/mohamed sliman/rambam_nifti_localizers'
EXPERIMENTS_DIR = BASE_DIR / 'experiments_height_pytorch'

# --- Hyperparameters ---
IMG_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 0  # Set to 0 for debugging, increase for speed on Linux
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Preprocessing ---
# Bone Window Settings (W:1800, L:400) -> Range: [-500, 1300]
WIN_MIN = 400 - (1800 // 2)
WIN_MAX = 400 + (1800 // 2)