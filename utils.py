import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path

from config import (
    EXCEL_PATH, NIFTI_ROOT, REQUIRED_COLUMNS,
    EXPERIMENTS_DIR, NUM_FOLDS, VAL_FRAC, RANDOM_SEED, HEIGHT_RANGE_EDGES_CM
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


def create_fold_splits_train_val_test(
    data_df: pd.DataFrame,
    num_folds: int = NUM_FOLDS,
    val_frac: float = VAL_FRAC,
    random_seed: int = RANDOM_SEED
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    Patient-level K-fold split where every fold has the same height distribution.

      • TEST: patients are sorted by height and taken in consecutive blocks of
        num_folds. Each block puts exactly one patient into every fold, so each
        fold gets one patient from every narrow height band. Inside a block, the
        patient with the most images goes to the fold with the fewest images so
        far, which keeps the image counts balanced too.
      • VAL: val_frac of all patients, taken from this fold's non-test patients:
        one from each equal slice of the height-sorted list, so validation also
        covers the whole height range.
      • TRAIN: all remaining patients.

    Returns:
      test_groups, val_groups, train_groups, all_patient_ids
    """
    # One row per patient: mean height and number of images
    patient_df = data_df.groupby('Patient_ID', as_index=False).agg(
        height_cm=('height_cm', 'mean'),
        n_images=('height_cm', 'size')
    )
    all_patient_ids = patient_df['Patient_ID'].to_numpy()

    n_patients = len(patient_df)
    if n_patients < num_folds:
        raise ValueError(f"Need at least {num_folds} patients for {num_folds} folds, got {n_patients}.")
    n_val = int(round(val_frac * n_patients))
    min_non_test = n_patients - int(np.ceil(n_patients / num_folds))
    if not 1 <= n_val < min_non_test:
        raise ValueError(f"val_frac={val_frac} gives {n_val} validation patients; "
                         f"must be between 1 and {min_non_test - 1}.")

    rng = np.random.default_rng(random_seed)

    # Sort by height; shuffling first breaks ties between equal heights at random
    patient_df = patient_df.iloc[rng.permutation(n_patients)]
    patient_df = patient_df.sort_values('height_cm', kind='stable').reset_index(drop=True)
    ids = patient_df['Patient_ID'].to_numpy()
    n_images = patient_df['n_images'].to_numpy()

    # ---- TEST: each block of num_folds consecutive heights gives one patient to every fold ----
    test_fold = np.empty(n_patients, dtype=int)
    fold_patients = np.zeros(num_folds, dtype=int)
    fold_images = np.zeros(num_folds, dtype=int)
    for start in range(0, n_patients, num_folds):
        block = np.arange(start, min(start + num_folds, n_patients))
        # Patients with the most images first ...
        block = block[np.argsort(-n_images[block], kind='stable')]
        # ... go to the folds with the fewest patients, then fewest images (random tie-break)
        fold_order = np.lexsort((rng.random(num_folds), fold_images, fold_patients))
        for i, f in zip(block, fold_order):
            test_fold[i] = f
            fold_patients[f] += 1
            fold_images[f] += n_images[i]

    test_groups, val_groups, train_groups = [], [], []
    for f in range(num_folds):
        # ---- VAL: one random patient from each height slice of the non-test patients ----
        non_test = np.flatnonzero(test_fold != f)  # still sorted by height
        bounds = np.round(np.linspace(0, len(non_test), n_val + 1)).astype(int)
        val_idx = [rng.choice(non_test[lo:hi]) for lo, hi in zip(bounds[:-1], bounds[1:])]

        # ---- TRAIN = complement ----
        train_mask = test_fold != f
        train_mask[val_idx] = False

        test_groups.append(np.array(sorted(ids[test_fold == f]), dtype=object))
        val_groups.append(np.array(sorted(ids[val_idx]), dtype=object))
        train_groups.append(np.array(sorted(ids[train_mask]), dtype=object))

    return test_groups, val_groups, train_groups, all_patient_ids


def assign_height_range(heights: pd.Series, edges: List[float] = HEIGHT_RANGE_EDGES_CM) -> pd.Series:
    """Bin heights (cm) into <e0, e0-e1, ..., >=eN (lower bound inclusive)."""
    labels = ([f"<{edges[0]:g}"] +
              [f"{lo:g}-{hi:g}" for lo, hi in zip(edges[:-1], edges[1:])] +
              [f">={edges[-1]:g}"])
    return pd.cut(heights, bins=[-np.inf, *edges, np.inf], labels=labels, right=False)


def summarize_fold_splits(
    data_df: pd.DataFrame,
    test_groups: List[np.ndarray],
    val_groups: List[np.ndarray],
    train_groups: List[np.ndarray]
) -> pd.DataFrame:
    """Per fold and subset: patient/image counts, height stats and patients per height range."""
    patient_heights = data_df.groupby('Patient_ID')['height_cm'].mean()
    images_per_patient = data_df.groupby('Patient_ID').size()

    rows = []
    for f in range(len(test_groups)):
        for subset, ids in (('Train', train_groups[f]), ('Val', val_groups[f]), ('Test', test_groups[f])):
            heights = patient_heights.loc[ids]
            row = {
                'Fold': f + 1,
                'Subset': subset,
                'Patients': len(ids),
                'Images': int(images_per_patient.loc[ids].sum()),
                'Height_Mean': heights.mean(),
                'Height_SD': heights.std(),
                'Height_Min': heights.min(),
                'Height_Max': heights.max(),
            }
            row.update(assign_height_range(heights).value_counts(sort=False).to_dict())
            rows.append(row)

    return pd.DataFrame(rows)





def get_fold_dataframes_explicit(
    data_df: pd.DataFrame,
    test_groups: List[np.ndarray],
    val_groups: List[np.ndarray],
    train_groups: List[np.ndarray],
    fold_idx: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Directly uses the patient IDs returned by create_fold_splits_train_val_test.
    """
    test_pats  = set(test_groups[fold_idx].tolist())
    val_pats   = set(val_groups[fold_idx].tolist())
    train_pats = set(train_groups[fold_idx].tolist())

    # Build dataframes
    train_df = data_df[data_df['Patient_ID'].isin(train_pats)].reset_index(drop=True)
    val_df   = data_df[data_df['Patient_ID'].isin(val_pats)].reset_index(drop=True)
    test_df  = data_df[data_df['Patient_ID'].isin(test_pats)].reset_index(drop=True)

    # Safety: ensure no leakage

    overlap = (
            (set(train_df['Patient_ID']) & set(val_df['Patient_ID'])) |
            (set(train_df['Patient_ID']) & set(test_df['Patient_ID'])) |
            (set(val_df['Patient_ID']) & set(test_df['Patient_ID']))
    )

    if overlap:
        raise RuntimeError(f"Data leakage detected: patient overlap across splits: {sorted(overlap)}")

    # Optional: summary
    print(f"\nFold {fold_idx + 1} Data Split:")
    print(f"  Train: {len(train_df)} rows ({train_df['Patient_ID'].nunique()} patients)")
    print(f"  Val:   {len(val_df)} rows ({val_df['Patient_ID'].nunique()} patients)")
    print(f"  Test:  {len(test_df)} rows ({test_df['Patient_ID'].nunique()} patients)")

    return train_df, val_df, test_df



def compute_height_range_errors(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Test error per true-height range, plus an 'All' row for everyone together.

    predictions_df: one row per image with Patient_ID, height_cm and Predicted_Height
    (as returned by save_fold_predictions; concatenate the folds for the pooled report).
    Bias_cm is the mean of (predicted - true): negative means height is underestimated.
    """
    errors = predictions_df['Predicted_Height'] - predictions_df['height_cm']
    height_ranges = assign_height_range(predictions_df['height_cm'])

    groups = [(label, height_ranges == label) for label in height_ranges.cat.categories]
    groups.append(('All', pd.Series(True, index=predictions_df.index)))

    rows = []
    for label, mask in groups:
        rows.append({
            'Height_Range': label,
            'Patients': predictions_df.loc[mask, 'Patient_ID'].nunique(),
            'Images': int(mask.sum()),
            'MAE_cm': errors[mask].abs().mean(),
            'Bias_cm': errors[mask].mean(),
        })

    return pd.DataFrame(rows)


def print_height_range_errors(range_errors: pd.DataFrame, title: str):
    print(f"\n{title}:")
    table = range_errors.to_string(index=False, float_format=lambda v: f"{v:.2f}", na_rep='-')
    for line in table.splitlines():
        print(f"  {line}")
    print("  (Bias = mean of predicted - true; negative = underestimated)")


def save_results_to_excel(
        all_results: list,
        fold_performance: list,
        output_path: str,
        height_range_errors: Optional[pd.DataFrame] = None
):
    results_df = pd.DataFrame(all_results)
    summary_df = pd.DataFrame({
        'Fold': range(1, len(fold_performance) + 1),
        'Test_MAE': fold_performance
    })

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Detailed_Logs', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        if height_range_errors is not None:
            height_range_errors.to_excel(writer, sheet_name='Height_Range_MAE', index=False)

    print(f"\nResults saved to '{output_path}'")
    print(f"Average TEST MAE: {np.mean(fold_performance):.2f} ± {np.std(fold_performance):.2f} cm")


def save_fold_predictions(
        test_df: pd.DataFrame,
        predictions: np.ndarray,
        fold_idx: int,
        output_dir: str = "experiments_height_pytorch"
) -> pd.DataFrame:
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

    return results_df
