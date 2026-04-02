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


from typing import Tuple, List, Dict, Optional

# Forced anchors for TEST per fold (required)
FORCED_TEST_PATIENTS_BY_FOLD = {
    0: 'C19',
    1: 'C22',
    2: 'C24',
    3: 'C38'
}

def _derive_forced_val_from_test(forced_test_by_fold: Dict[int, str]) -> Dict[int, str]:
    """
    If user doesn't pass a val mapping, derive one so that:
      - Each anchor appears exactly once as a val anchor across folds.
      - The val anchor for fold f is different from the test anchor of fold f.
    Strategy: rotate the test anchors by +1.
    """
    folds = sorted(forced_test_by_fold.keys())
    ordered_anchors = [forced_test_by_fold[f] for f in folds]
    rotated = ordered_anchors[1:] + ordered_anchors[:1]
    return {f: v for f, v in zip(folds, rotated)}





def _derive_forced_val_from_test(forced_test_by_fold: Dict[int, str]) -> Dict[int, str]:
    """Rotate test anchors by +1 to auto-derive validation anchors."""
    folds = sorted(forced_test_by_fold.keys())
    ordered_anchors = [forced_test_by_fold[f] for f in folds]
    rotated = ordered_anchors[1:] + ordered_anchors[:1]
    return {f: v for f, v in zip(folds, rotated)}

def create_fold_splits_train_val_test(
    data_df: pd.DataFrame,
    num_folds: int = 4,
    forced_test_patients_by_fold: Dict[int, str] = None,
    forced_val_patients_by_fold: Optional[Dict[int, str]] = None,
    test_frac: float = 0.25,
    val_frac: float = 0.20,
    random_seed: int = 42,
    n_strata_bins: int = 4
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    Build 4 folds with:
      • Exactly one forced anchor in TEST for that fold.
      • Exactly one (different) forced anchor in VAL for that fold.
      • The remaining two anchors in TRAIN for that fold.
      • Non-anchors stratified by height and balanced to reach target sizes.

    Returns:
      test_groups, val_groups, train_groups, all_patient_ids
    """
    if num_folds != 4:
        raise ValueError("This project currently expects NUM_FOLDS=4.")
    if forced_test_patients_by_fold is None:
        raise ValueError("forced_test_patients_by_fold is required.")

    folds = list(range(num_folds))
    if sorted(forced_test_patients_by_fold.keys()) != folds:
        raise ValueError("forced_test_patients_by_fold must have keys 0..num_folds-1.")

    if forced_val_patients_by_fold is None:
        forced_val_patients_by_fold = _derive_forced_val_from_test(forced_test_patients_by_fold)
    if sorted(forced_val_patients_by_fold.keys()) != folds:
        raise ValueError("forced_val_patients_by_fold must have keys 0..num_folds-1.")
    for f in folds:
        if forced_val_patients_by_fold[f] == forced_test_patients_by_fold[f]:
            raise ValueError(f"Fold {f}: test and val anchors must be different.")

    # Aggregate to per-patient mean height
    patient_df = data_df.groupby('Patient_ID', as_index=False)['height_cm'].mean()
    all_patient_ids = patient_df['Patient_ID'].to_numpy()

    # Targets
    total_patients = len(all_patient_ids)
    target_test = [int(round(test_frac * total_patients)) for _ in folds]  # ~10–11 for 42
    target_val  = [int(round(val_frac  * total_patients)) for _ in folds]  # ~8–9  for 42

    anchors_all = set(list(forced_test_patients_by_fold.values()) +
                      list(forced_val_patients_by_fold.values()))
    rng = np.random.default_rng(random_seed)

    # ---- TEST: seed forced anchor per fold, then fill with non-anchors ----
    test_groups: List[List[object]] = [[forced_test_patients_by_fold[f]] for f in folds]
    non_anchor_df = patient_df[~patient_df['Patient_ID'].isin(anchors_all)].copy()
    if not non_anchor_df.empty:
        n_bins_eff = min(n_strata_bins, non_anchor_df['height_cm'].nunique())
        if n_bins_eff > 1:
            non_anchor_df['height_bin'] = pd.qcut(
                non_anchor_df['height_cm'],
                q=n_bins_eff, labels=False, duplicates='drop'
            )
        else:
            non_anchor_df['height_bin'] = 0

        for _, grp in non_anchor_df.groupby('height_bin'):
            ids = grp['Patient_ID'].tolist()
            rng.shuffle(ids)
            for pid in ids:
                # Prefer fold with smallest test size if still under target
                target_fold = int(np.argmin([len(g) for g in test_groups]))
                if len(test_groups[target_fold]) < target_test[target_fold]:
                    test_groups[target_fold].append(pid)
                else:
                    # If targets met, still assign to the smallest to finish (rare)
                    alt_fold = int(np.argmin([len(g) for g in test_groups]))
                    test_groups[alt_fold].append(pid)

    # ---- VAL: per fold, seed forced val anchor, then fill up to target with non-anchors not in this fold’s TEST ----
    val_groups: List[List[object]] = [[forced_val_patients_by_fold[f]] for f in folds]
    for f in folds:
        test_set_f = set(test_groups[f])
        # Eligible: non-anchors NOT in this fold's TEST
        eligible_df = patient_df[
            (~patient_df['Patient_ID'].isin(anchors_all)) &
            (~patient_df['Patient_ID'].isin(test_set_f))
        ].copy()

        if not eligible_df.empty:
            n_bins_eff = min(n_strata_bins, eligible_df['height_cm'].nunique())
            if n_bins_eff > 1:
                eligible_df['height_bin'] = pd.qcut(
                    eligible_df['height_cm'],
                    q=n_bins_eff, labels=False, duplicates='drop'
                )
            else:
                eligible_df['height_bin'] = 0

            needed = max(target_val[f] - len(val_groups[f]), 0)

            # Greedy bin-cycled pick until target reached
            bin_to_ids = {b: grp['Patient_ID'].tolist() for b, grp in eligible_df.groupby('height_bin')}
            for b in bin_to_ids:
                rng.shuffle(bin_to_ids[b])

            bins_order = list(bin_to_ids.keys())
            idx_per_bin = {b: 0 for b in bins_order}
            picked = 0
            while picked < needed and bins_order:
                exhausted = []
                for b in bins_order:
                    ids_list = bin_to_ids[b]
                    i = idx_per_bin[b]
                    if i < len(ids_list):
                        val_groups[f].append(ids_list[i])
                        idx_per_bin[b] += 1
                        picked += 1
                        if picked >= needed:
                            break
                    else:
                        exhausted.append(b)
                if exhausted:
                    bins_order = [b for b in bins_order if b not in exhausted]

    # ---- TRAIN = complement per fold ----
    all_set = set(all_patient_ids)
    test_groups = [np.array(sorted(g), dtype=object) for g in test_groups]
    val_groups  = [np.array(sorted(g), dtype=object) for g in val_groups]
    train_groups = []
    for f in folds:
        train_f = np.array(sorted(all_set - set(test_groups[f]) - set(val_groups[f])), dtype=object)
        train_groups.append(train_f)

    # Anchors sanity: 1 in test, 1 in val, other 2 in train
    for f in folds:
        assert forced_test_patients_by_fold[f] in set(test_groups[f])
        assert forced_val_patients_by_fold[f]  in set(val_groups[f])
        others = anchors_all - {forced_test_patients_by_fold[f], forced_val_patients_by_fold[f]}
        assert others.issubset(set(train_groups[f]))

    return test_groups, val_groups, train_groups, all_patient_ids





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
        set(train_df['Patient_ID']) & set(val_df['Patient_ID'])
    ) | (
        set(train_df['Patient_ID']) & set(test_df['Patient_ID'])
    ) | (
        set(val_df['Patient_ID']) & set(test_df['Patient_ID'])
    )
    if overlap:
        raise RuntimeError(f"Data leakage detected: patient overlap across splits: {sorted(overlap)}")

    # Optional: summary
    print(f"\nFold {fold_idx + 1} Data Split:")
    print(f"  Train: {len(train_df)} rows ({train_df['Patient_ID'].nunique()} patients)")
    print(f"  Val:   {len(val_df)} rows ({val_df['Patient_ID'].nunique()} patients)")
    print(f"  Test:  {len(test_df)} rows ({test_df['Patient_ID'].nunique()} patients)")

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
