import os
import json
import glob
import numpy as np
import nibabel as nib
import pandas as pd
from datetime import datetime

# --- CONFIGURATION ---
# Replace this with the actual path to your 'rambam_nifti_localizers' folder
ROOT_DIR = r"C:\Users\Lab2\Desktop\mohamed sliman\rambam_nifti_localizers"
OUTPUT_EXCEL = "Rambam_Localizer_Metadata.xlsx"


def get_nifti_metadata(nii_path):
    """Extracts physical data from the NIfTI header and image data."""
    try:
        nii = nib.load(nii_path)
        img_data = nii.get_fdata()
        header = nii.header

        # 1. Dimensions & Spacing
        dims = header.get_data_shape()
        zooms = header.get_zooms()  # Voxel sizes (x, y, z)

        # 2. Hounsfield Unit (HU) Statistics
        # We use standard numpy functions to get the range
        min_hu = np.min(img_data)
        max_hu = np.max(img_data)
        mean_hu = np.mean(img_data)

        return {
            "Dim_X": dims[0],
            "Dim_Y": dims[1],
            "Dim_Z": dims[2] if len(dims) > 2 else 1,
            "Spacing_X": round(zooms[0], 3),
            "Spacing_Y": round(zooms[1], 3),
            "Spacing_Z": round(zooms[2], 3) if len(zooms) > 2 else 0,
            "Min_HU": round(min_hu, 2),
            "Max_HU": round(max_hu, 2),
            "Mean_HU": round(mean_hu, 2),
            "Data_Type": header.get_data_dtype().name
        }
    except Exception as e:
        print(f"Error reading NIfTI {nii_path}: {e}")
        return {}


def get_json_metadata(json_path):
    """Extracts scanner info from the JSON sidecar file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Common keys in medical JSONs (e.g., from dcm2niix)
        # We use .get() to avoid crashing if a key is missing
        return {
            "Scanner_Manufacturer": data.get("Manufacturer", "Unknown"),
            "Scanner_Model": data.get("ManufacturerModelName", "Unknown"),
            "Modality": data.get("Modality", "Unknown"),
            "KVP": data.get("KVP", "Unknown"),
            "SliceThickness": data.get("SliceThickness", "Unknown"),
            "PixelSpacing_JSON": str(data.get("PixelSpacing", "Unknown")),  # Keep as string to avoid formatting issues
            "SeriesDescription": data.get("SeriesDescription", "Unknown"),
            "ProtocolName": data.get("ProtocolName", "Unknown")
        }
    except Exception as e:
        print(f"Error reading JSON {json_path}: {e}")
        return {}


def main():
    print(f"Scanning folder: {ROOT_DIR} ...")

    records = []

    # Recursively find all .nii.gz files
    # The pattern **/*.nii.gz searches all subdirectories
    nifti_files = glob.glob(os.path.join(ROOT_DIR, "**", "*.nii.gz"), recursive=True)

    print(f"Found {len(nifti_files)} NIfTI files. Processing...")

    for nii_path in nifti_files:
        # Construct expected JSON path (same name, but .json extension)
        # This handles .nii.gz -> .json (removing .nii.gz first)
        base_name = nii_path.replace(".nii.gz", "")
        json_path = base_name + ".json"

        # Basic File Info
        folder_name = os.path.basename(os.path.dirname(nii_path))  # e.g., 000029A2
        file_name = os.path.basename(nii_path)

        # 1. Get NIfTI Info
        nii_meta = get_nifti_metadata(nii_path)

        # 2. Get JSON Info (if it exists)
        json_meta = {}
        if os.path.exists(json_path):
            json_meta = get_json_metadata(json_path)
        else:
            json_meta = {"Scanner_Manufacturer": "JSON Missing"}

        # 3. Combine Metadata
        row = {
            "Folder_ID": folder_name,
            "File_Name": file_name,
            "Full_Path": nii_path,
            **nii_meta,  # Unpack NIfTI dictionary
            **json_meta  # Unpack JSON dictionary
        }

        records.append(row)

        if len(records) % 100 == 0:
            print(f"Processed {len(records)} images...")

    # Create DataFrame and Save
    df = pd.DataFrame(records)

    # Reorder columns to put interesting stuff first
    cols = ["Folder_ID", "Scanner_Manufacturer", "Scanner_Model", "Min_HU", "Max_HU",
            "Dim_X", "Dim_Y", "Spacing_X", "Spacing_Y", "File_Name"]
    # Add whatever remaining columns exist
    cols += [c for c in df.columns if c not in cols]
    df = df[cols]

    print(f"Saving to {OUTPUT_EXCEL}...")
    df.to_excel(OUTPUT_EXCEL, index=False)
    print("Done!")


if __name__ == "__main__":
    main()