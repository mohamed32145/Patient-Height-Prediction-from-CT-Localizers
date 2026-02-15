
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


from config import (
    EXCEL_PATH, NIFTI_ROOT, REQUIRED_COLUMNS,
    EXPERIMENTS_DIR, RANDOM_SEED
)


def load_and_validate_dataframe() -> pd.DataFrame:
    """
    Load Excel file and validate required columns.

    Returns:
        pd.DataFrame: Validated dataframe with required columns
    """
    df = pd.read_excel(EXCEL_PATH, engine='openpyxl')
    df.columns = [str(c).strip() for c in df.columns]

    # Validate required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Sanitize Height (numeric, in cm)
    df['Height'] = pd.to_numeric(df['Height'], errors='coerce')
    # Drop rows without height
    df = df.dropna(subset=['Height'])

    return df


def resolve_nifti_dir(localizer_dir_str: str) -> Optional[Path]:
    """
    Resolve NIfTI directory path from various formats.

    Args:
        localizer_dir_str: Path string from Excel file

    Returns:
        Path object if directory exists, None otherwise
    """
    if not isinstance(localizer_dir_str, str) or not localizer_dir_str.strip():
        return None

    marker = 'rambam_nifti_localizers'
    s = localizer_dir_str.strip()

    if marker in s:
        tail = s.split(marker, maxsplit=1)[-1].strip('\\/ ')
        rel = Path(tail.replace('\\', '/'))
        candidate = NIFTI_ROOT / rel
    else:
        # Normalize separators; decide whether it's absolute or relative
        normalized = Path(s.replace('\\', '/'))
        candidate = normalized if normalized.is_absolute() else (NIFTI_ROOT / normalized)

    # Return only existing directories
    return candidate if candidate.exists() and candidate.is_dir() else None


def pick_nifti_file(nifti_dir: Path) -> Optional[Path]:
    """
    Find first NIfTI file in directory.

    Args:
        nifti_dir: Directory containing NIfTI files

    Returns:
        Path to first .nii or .nii.gz file found, or None
    """
    files = sorted(nifti_dir.glob('*.nii')) + sorted(nifti_dir.glob('*.nii.gz'))
    return files[0] if files else None


def prepare_dataset() -> pd.DataFrame:
    """
    Load Excel, resolve NIfTI paths, and prepare final dataset.

    Returns:
        pd.DataFrame: Dataset with columns ['Patient_ID', 'nifti_path', 'height_cm']
    """
    df = load_and_validate_dataframe()

    rows = []
    skipped_unresolved_dir = 0
    skipped_no_files = 0

    for _, r in df.iterrows():
        pid = str(r['Patient_ID']).strip()
        height_cm = float(r['Height'])
        d = resolve_nifti_dir(r['Localizer_Path_NIfTI'])
        if d is None:
            skipped_unresolved_dir += 1
            continue
        f = pick_nifti_file(d)
        if f is None:
            skipped_no_files += 1
            continue
        rows.append({
            'Patient_ID': pid,
            'nifti_path': str(f),
            'height_cm': height_cm
        })

    data_df = pd.DataFrame(rows)

    print(f"Resolved NIfTI files for {len(data_df)} rows")
    print(f"Skipped unresolved dir: {skipped_unresolved_dir}")
    print(f"Skipped empty NIfTI dir: {skipped_no_files}")
    print(f"Total Patients: {data_df['Patient_ID'].nunique()}")

    return data_df


def create_fold_splits(
        data_df: pd.DataFrame,
        num_folds: int = 4
) -> Tuple[list, list]:
    """
    Create patient-level fold splits for rotating cross-validation.

    Args:
        data_df: DataFrame with patient data
        num_folds: Number of folds for cross-validation

    Returns:
        Tuple of (patient_groups, patient_ids)
    """
    # Get unique patients and shuffle them
    patient_ids = data_df['Patient_ID'].unique()
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(patient_ids)

    # Split patients into N roughly equal groups
    patient_groups = np.array_split(patient_ids, num_folds)

    print(f"\nFold Split Summary:")
    print(f"Total Patients: {len(patient_ids)}")
    print(f"Patients per fold: {[len(g) for g in patient_groups]}")

    return patient_groups, patient_ids


def get_fold_dataframes(
        data_df: pd.DataFrame,
        patient_groups: list,
        fold_idx: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Get train/val/test dataframes for a specific fold in rotating CV.

    The rotation scheme:
    - Group i         -> TEST (Held out completely)
    - Group (i+1)%4   -> VALIDATION (Used for model tuning)
    - Remaining 2     -> TRAIN

    Args:
        data_df: Full dataset
        patient_groups: List of patient ID arrays for each group
        fold_idx: Current fold index (0-based)

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    num_folds = len(patient_groups)

    test_pats = patient_groups[fold_idx]
    val_pats = patient_groups[(fold_idx + 1) % num_folds]

    # Concatenate the remaining groups for training
    train_pats = np.concatenate([
        patient_groups[(fold_idx + 2) % num_folds],
        patient_groups[(fold_idx + 3) % num_folds]
    ])

    # Filter DataFrame
    train_df = data_df[data_df['Patient_ID'].isin(train_pats)].reset_index(drop=True)
    val_df = data_df[data_df['Patient_ID'].isin(val_pats)].reset_index(drop=True)
    test_df = data_df[data_df['Patient_ID'].isin(test_pats)].reset_index(drop=True)

    print(f"\nFold {fold_idx + 1} Data Split:")
    print(f"  Train: {len(train_df)} images ({train_df['Patient_ID'].nunique()} patients)")
    print(f"  Val:   {len(val_df)} images ({val_df['Patient_ID'].nunique()} patients)")
    print(f"  Test:  {len(test_df)} images ({test_df['Patient_ID'].nunique()} patients)")

    return train_df, val_df, test_df


def save_results_to_excel(all_results: list, fold_performance: list, output_path: str):
    """
    Save training results to Excel file.

    Args:
        all_results: List of dictionaries with detailed training logs
        fold_performance: List of test MAE scores for each fold
        output_path: Path to save Excel file
    """
    results_df = pd.DataFrame(all_results)
    summary_df = pd.DataFrame({
        'Fold': range(1, len(fold_performance) + 1),
        'Test_MAE': fold_performance
    })

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Detailed_Logs', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

    print(f"\nResults saved to '{output_path}'")
    print(f"Average TEST MAE: {np.mean(fold_performance):.2f} ± {np.std(fold_performance):.2f} cm")