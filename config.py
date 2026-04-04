
from pathlib import Path



# ============================================================================
# PATH CONFIGURATION
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
EXCEL_PATH = BASE_DIR / 'C:/Users/Lab2/Desktop/mohamed sliman/Patients_list_body_CT_localizers_expanded.xlsx'
NIFTI_ROOT = BASE_DIR / 'C:/Users/Lab2/Desktop/mohamed sliman/rambam_nifti_localizers'
EXPERIMENTS_DIR = BASE_DIR / 'experiments_height_pytorch'

# Model weights path
WEIGHTS_PATH = "C:/Users/Lab2/Desktop/mohamed sliman/RadImageNet_pytorch/ResNet50.pt"



# ============================================================================
# DATA CONFIGURATION
# ============================================================================
# Required columns in Excel file
REQUIRED_COLUMNS = ['Patient_ID', 'Height', 'Localizer_Path_NIfTI']

# Image processing parameters
IMG_SIZE = 384
TARGET_PIXEL_SPACING_MM = 1.0

WIN_MIN = 400 - (1800 // 2)  # -500 for bone windowing
WIN_MAX = 400 + (1800 // 2)  # 1300 for bone windowing

# Spine cropping parameters


# Lowered slightly to 50 to detect bodies in 8-bit images more reliably
THRESHOLD_VALUE = 50


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# EfficientNetV2 feature dimension (for efficientnet_v2_s)
BACKBONE_NAME = 'efficientnet_v2_s'
BACKBONE_FEATURE_DIM = 1280
USE_IMAGENET_PRETRAINED = True

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
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
LOSS_NAME = 'mse'

# LR scheduling
USE_COSINE_ANNEALING = True
COSINE_T_MAX = NUM_EPOCHS
COSINE_ETA_MIN = 1e-6

# Device configuration
DEVICE = 'cuda'

# Logging frequency (print every N epochs)
LOG_FREQUENCY = 5

# Forced fold assignment (strict patient-level CV anchors)
FORCED_TEST_PATIENTS_BY_FOLD = {
    0: 'C19',
    1: 'C22',
    2: 'C24',
    3: 'C38'
}

# ============================================================================
# AUGMENTATION CONFIGURATION
# ============================================================================
# Training augmentations
AUG_HORIZONTAL_FLIP_PROB = 0.3
AUG_SHIFT_LIMIT = 0.1
AUG_SCALE_LIMIT = 0.0
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
