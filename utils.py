import pandas as pd
import numpy as np
from typing import Optional, Tuple
from pathlib import Path

from config import (
    EXCEL_PATH, NIFTI_ROOT, REQUIRED_COLUMNS,
    EXPERIMENTS_DIR, FORCED_TEST_PATIENTS_BY_FOLD, RANDOM_SEED
)


def load_and_validate_dataframe() -> pd.DataFrame:
    df = pd.read_excel(EXCEL_PATH, engine='openpyxl')
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df['Height'] = pd.to_numeric(df['Height'], errors='coerce')
    df = df.dropna(subset=['Height'])
    return df


def resolve_nifti_dir(localizer_dir_str: str) -> Optional[Path]:
    if not isinstance(localizer_dir_str, str) or not localizer_dir_str.strip():
        return None

    marker = 'rambam_nifti_localizers'
    s = localizer_dir_str.strip()

    if marker in s:
        tail = s.split(marker, maxsplit=1)[-1].strip('\\/ ')
        rel = Path(tail.replace('\\', '/'))
        candidate = NIFTI_ROOT / rel
    else:
        normalized = Path(s.replace('\\', '/'))
        candidate = normalized if normalized.is_absolute() else (NIFTI_ROOT / normalized)

    return candidate if candidate.exists() and candidate.is_dir() else None


def pick_nifti_file(nifti_dir: Path) -> Optional[Path]:
    files = sorted(nifti_dir.glob('*.nii')) + sorted(nifti_dir.glob('*.nii.gz'))
    return files[0] if files else None


def prepare_dataset() -> pd.DataFrame:
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


def create_fold_splits(data_df: pd.DataFrame, num_folds: int = 4) -> Tuple[list, list]:
    """
    Strict patient-level split with forced anchors and approximate stratification by height.
    """
    if num_folds != 4:
        raise ValueError("This project currently expects NUM_FOLDS=4.")

    patient_df = data_df.groupby('Patient_ID', as_index=False)['height_cm'].mean()
    patient_ids = patient_df['Patient_ID'].to_numpy()

    # Initialize folds with required anchor patients
    patient_groups = [[] for _ in range(num_folds)]
    assigned = set()

    for fold_idx, patient_id in FORCED_TEST_PATIENTS_BY_FOLD.items():
        if patient_id in set(patient_ids):
            patient_groups[fold_idx].append(patient_id)
            assigned.add(patient_id)
        else:
            print(f"Warning: forced patient {patient_id} not found in dataset.")

    # Stratify remaining patients using height quantiles then greedy balance
    remaining = patient_df[~patient_df['Patient_ID'].isin(assigned)].copy()
    if not remaining.empty:
        n_bins = min(4, remaining['height_cm'].nunique())
        if n_bins > 1:
            remaining['height_bin'] = pd.qcut(
                remaining['height_cm'],
                q=n_bins,
                labels=False,
                duplicates='drop'
            )
        else:
            remaining['height_bin'] = 0

        rng = np.random.default_rng(RANDOM_SEED)
        for _, group in remaining.groupby('height_bin'):
            ids = group['Patient_ID'].tolist()
            rng.shuffle(ids)
            for pid in ids:
                target_fold = int(np.argmin([len(g) for g in patient_groups]))
                patient_groups[target_fold].append(pid)

    patient_groups = [np.array(g, dtype=object) for g in patient_groups]

    print("\nFold Split Summary:")
    print(f"Total Patients: {len(patient_ids)}")
    print(f"Patients per fold: {[len(g) for g in patient_groups]}")
    for fold_idx, group in enumerate(patient_groups):
        print(f"  Fold {fold_idx + 1} forced anchor: {FORCED_TEST_PATIENTS_BY_FOLD.get(fold_idx)}")
        print(f"    Patients: {sorted(group.tolist())}")

    return patient_groups, patient_ids


def get_fold_dataframes(data_df: pd.DataFrame, patient_groups: list, fold_idx: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    num_folds = len(patient_groups)

    test_pats = patient_groups[fold_idx]
    val_pats = patient_groups[(fold_idx + 1) % num_folds]
    train_pats = np.concatenate([
        patient_groups[(fold_idx + 2) % num_folds],
        patient_groups[(fold_idx + 3) % num_folds]
    ])

    train_df = data_df[data_df['Patient_ID'].isin(train_pats)].reset_index(drop=True)
    val_df = data_df[data_df['Patient_ID'].isin(val_pats)].reset_index(drop=True)
    test_df = data_df[data_df['Patient_ID'].isin(test_pats)].reset_index(drop=True)

    print(f"\nFold {fold_idx + 1} Data Split:")
    print(f"  Train: {len(train_df)} images ({train_df['Patient_ID'].nunique()} patients)")
    print(f"  Val:   {len(val_df)} images ({val_df['Patient_ID'].nunique()} patients)")
    print(f"  Test:  {len(test_df)} images ({test_df['Patient_ID'].nunique()} patients)")

    overlap = (
        set(train_df['Patient_ID']) & set(val_df['Patient_ID'])
    ) | (
        set(train_df['Patient_ID']) & set(test_df['Patient_ID'])
    ) | (
        set(val_df['Patient_ID']) & set(test_df['Patient_ID'])
    )
    if overlap:
        raise RuntimeError(f"Data leakage detected: patient overlap across splits: {sorted(overlap)}")

    return train_df, val_df, test_df


def save_results_to_excel(all_results: list, fold_performance: list, output_path: str):
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


def save_fold_predictions(
        test_df: pd.DataFrame,
        predictions: np.ndarray,
        fold_idx: int,
        output_dir: str = "experiments_height_pytorch"
):
    results_df = test_df.copy()
    results_df['Predicted_Height'] = predictions.flatten()

    true_label_col = 'Height' if 'Height' in results_df.columns else 'height_cm'
    results_df['Absolute_Error'] = np.abs(results_df['Predicted_Height'] - results_df[true_label_col])

    results_df = results_df.sort_values(by='Absolute_Error', ascending=False)

    display_cols = ['Patient_ID', true_label_col, 'Predicted_Height', 'Absolute_Error']
    if 'Localizer_Path_NIfTI' in results_df.columns:
        display_cols.append('Localizer_Path_NIfTI')
    elif 'nifti_path' in results_df.columns:
        display_cols.append('nifti_path')

    results_df = results_df[display_cols]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"fold_{fold_idx + 1}_patient_predictions.csv"

    results_df.to_csv(out_file, index=False)
    print(f"  -> Saved patient-level predictions to {out_file.name}")
