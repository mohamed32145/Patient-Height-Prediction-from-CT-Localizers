
from pathlib import Path



# ============================================================================
# PATH CONFIGURATION
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
EXCEL_PATH = BASE_DIR / 'C:/Users/Lab2/Desktop/mohamed sliman/Patients_list_body_CT_localizers.xlsx'
NIFTI_ROOT = BASE_DIR / 'C:/Users/Lab2/Desktop/mohamed sliman/rambam_nifti_localizers'
EXPERIMENTS_DIR = BASE_DIR / 'experiments_height_pytorch'

# Model weights path
WEIGHTS_PATH = "C:/Users/Lab2/Desktop/mohamed sliman/ResNet50.pt"

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
# Required columns in Excel file
REQUIRED_COLUMNS = ['Patient_ID', 'Height', 'Localizer_Path_NIfTI']

# Image processing parameters
IMG_SIZE = 256
WIN_MIN = 400 - (1800 // 2)  # -500 for bone windowing
WIN_MAX = 400 + (1800 // 2)  # 1300 for bone windowing

# Spine cropping parameters
CROP_WIDTH = 170  # Narrow crop centered on spine
THRESHOLD_VALUE = 60  # Threshold for segmentation (higher = ignores bed)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# ResNet50 feature dimension
RESNET_FEATURE_DIM = 2048

# Metadata (pixel spacing) dimension
METADATA_DIM = 2

# Hidden layer dimensions
METADATA_HIDDEN_DIM = 8
REGRESSOR_HIDDEN_DIM = 512

# Dropout rate
DROPOUT_RATE = 0.3

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
# Cross-validation settings
NUM_FOLDS = 4
RANDOM_SEED = 42

# Training hyperparameters
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Device configuration
DEVICE = 'cuda'

# Logging frequency (print every N epochs)
LOG_FREQUENCY = 5

# ============================================================================
# AUGMENTATION CONFIGURATION
# ============================================================================
# Training augmentations
AUG_HORIZONTAL_FLIP_PROB = 0.3
AUG_SHIFT_LIMIT = 0.1
AUG_SCALE_LIMIT = 0.15
AUG_ROTATE_LIMIT = 10
AUG_SHIFT_SCALE_ROTATE_PROB = 0.8
AUG_BRIGHTNESS_CONTRAST_PROB = 0.2

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================
# Results file
RESULTS_EXCEL_PATH = 'training_results_rotating.xlsx'

# Model checkpoint naming
MODEL_CHECKPOINT_PATTERN = 'height_model_fold_{fold}.pth'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def setup_directories():
    """Create necessary directories if they don't exist"""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

def get_device():
    """Get the appropriate device for training"""
    import torch
    return torch.device(DEVICE if torch.cuda.is_available() else "cpu")