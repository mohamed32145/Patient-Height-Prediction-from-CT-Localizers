from pathlib import Path

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
EXCEL_PATH = BASE_DIR / 'C:/Users/Lab2/Desktop/mohamed sliman/Patients_list_body_CT_localizers_expanded.xlsx'
NIFTI_ROOT = BASE_DIR / 'C:/Users/Lab2/Desktop/mohamed sliman/rambam_nifti_localizers'
EXPERIMENTS_DIR = BASE_DIR / 'experiments_height_pytorch'

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
REQUIRED_COLUMNS = ['Patient_ID', 'Height', 'Localizer_Path_NIfTI']

# Spatial normalization target after physical 1mm resampling
IMG_HEIGHT = 768
IMG_WIDTH = 512
TARGET_PIXEL_SPACING_MM = 1.0

# HU clipping and linear rescaling to uint8
HU_MIN = -256
HU_MAX = 1024

# keep localizer orientation policy simple: only HFS is allowed
ALLOWED_PATIENT_POSITION = 'HFS'

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# evaluated options: efficientnet_v2_s, resnet18, resnet34
BACKBONE_NAME = 'efficientnet_v2_s'
USE_IMAGENET_PRETRAINED = True

# three hidden layers for regression head
HEAD_HIDDEN_DIMS = [1024, 128, 8]
DROPOUT_RATE = 0.1
FREEZE_STAGES = 5

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
NUM_FOLDS = 4
RANDOM_SEED = 42

BATCH_SIZE = 16
NUM_EPOCHS = 51
LEARNING_RATE = 2.5e-4
WEIGHT_DECAY = 0.01
ADAMW_BETAS = (0.9, 0.999)
LOSS_NAME = 'mse'

# Step LR schedule: multiply LR by 0.8 every 21 epochs
LR_STEP_SIZE = 21
LR_GAMMA = 0.8

DEVICE = 'cuda'
LOG_FREQUENCY = 5

FORCED_TEST_PATIENTS_BY_FOLD = {
    0: 'C19',
    1: 'C22',
    2: 'C24',
    3: 'C38'
}

FORCED_VAL_PATIENTS_BY_FOLD = {
    0: 'C22',
    1: 'C24',
    2: 'C38',
    3: 'C19'
}

# ============================================================================
# AUGMENTATION CONFIGURATION
# ============================================================================
AUG_ROTATE_LIMIT = 5
AUG_BRIGHTNESS_LIMIT = 0.1
AUG_CONTRAST_LIMIT = 0.1
AUG_ERASE_MAX_FRACTION = 0.15
AUGMENTATION_PROBABILITY = 0.8

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================
RESULTS_EXCEL_PATH = 'training_results_rotating.xlsx'
MODEL_CHECKPOINT_PATTERN = 'height_model_fold_{fold}.pth'


def setup_directories():
    """Create necessary directories if they don't exist."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


def get_device():
    """Get the appropriate device for training."""
    import torch
    return torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
