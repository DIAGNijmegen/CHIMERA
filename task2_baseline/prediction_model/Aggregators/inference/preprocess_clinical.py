#!/usr/bin/env python3
"""
Training-time JSON processor for CHIMERA Task 2.

- Reads raw clinical JSON(s)
- Fits categorical encoders and numeric scaler
- Derives sample_id from filename: everything before '_CD' (case-insensitive)
- Saves processed features CSV + processor PKL
"""

import argparse
import json
from pathlib import Path
import pickle
from typing import Dict, List, Any
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


# =========================
# Config
# =========================
CATEGORICAL_COLS = [
    "sex", "smoking", "tumor", "stage", "substage",
    "grade", "reTUR", "LVI", "variant", "EORTC"
]
NUMERICAL_COLS = ["age", "no_instillations"]
ID_COL = "sample_id"
LABEL_COL = "BRS"
MISSING_CODE = -1


# =========================
# Helpers
# =========================
def extract_case_id(json_path: Path) -> str:
    """Take everything before '_CD' (case-insensitive) in filename, else full stem."""
    stem = json_path.stem
    parts = stem.split("_")
    for i, p in enumerate(parts):
        if p.lower() == "cd":
            return "_".join(parts[:i])
    return stem


def is_missing(v: Any) -> bool:
    return pd.isna(v) or v == -1 or v == "-1" or (isinstance(v, str) and v.strip() == "")


def read_json_rows(paths: List[Path]) -> pd.DataFrame:
    rows = []
    for p in sorted(paths):
        with open(p, "r") as f:
            data = json.load(f)
        sid = extract_case_id(p)
        if isinstance(data, dict):
            rows.append({**data, ID_COL: sid})
        elif isinstance(data, list):
            for i, d in enumerate(data):
                if not isinstance(d, dict):
                    raise ValueError(f"{p} contains non-dict list items")
                rows.append({**d, ID_COL: f"{sid}__r{i}"})
        else:
            raise ValueError(f"{p} unsupported JSON structure")
    return pd.DataFrame(rows)


# =========================
# Main processing
# =========================
def process_and_save(json_paths: List[Path], output_dir: Path, keep_label: bool):
    df = read_json_rows(json_paths)
    df.columns = df.columns.str.strip()

    # Check required columns
    for col in CATEGORICAL_COLS + NUMERICAL_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Prepare features
    X = df[CATEGORICAL_COLS + NUMERICAL_COLS].copy()

    # Encode categoricals
    encoders: Dict[str, LabelEncoder] = {}
    category_maps: Dict[str, Dict[str, int]] = {}
    for col in CATEGORICAL_COLS:
        tr_raw = X[col].astype(object)
        tr_non_missing = tr_raw[~tr_raw.map(is_missing)].astype(str)
        le = LabelEncoder()
        if len(tr_non_missing) > 0:
            le.fit(tr_non_missing)
            mapping = {cls: int(i) for i, cls in enumerate(le.classes_)}
        else:
            le.classes_ = np.array([], dtype=object)
            mapping = {}
        def encode_value(v):
            if is_missing(v):
                return MISSING_CODE
            return mapping.get(str(v), MISSING_CODE)
        X[col] = tr_raw.map(encode_value).astype(int)
        encoders[col] = le
        category_maps[col] = mapping

    # Scale numerics
    X[NUMERICAL_COLS] = X[NUMERICAL_COLS].apply(pd.to_numeric, errors="coerce")
    if X[NUMERICAL_COLS].isna().any().any():
        X[NUMERICAL_COLS] = X[NUMERICAL_COLS].fillna(X[NUMERICAL_COLS].mean())
    scaler = StandardScaler()
    X[NUMERICAL_COLS] = scaler.fit_transform(X[NUMERICAL_COLS])

    # Reattach sample_id
    X[ID_COL] = df[ID_COL]

    # Optionally reattach label
    if keep_label and LABEL_COL in df.columns:
        X[LABEL_COL] = df[LABEL_COL]

    # Save CSV
    output_csv = output_dir / "processed_clinical_train.csv"
    X.to_csv(output_csv, index=False)

    # Save PKL
    processor = {
        "categorical_cols": CATEGORICAL_COLS,
        "numerical_cols": NUMERICAL_COLS,
        "missing_code": MISSING_CODE,
        "encoders": encoders,
        "category_maps": category_maps,
        "scaler": scaler,
        "id_col": ID_COL,
        "label_col": LABEL_COL,
    }
    output_pkl = output_dir / "clinical_processor_json.pkl"
    with open(output_pkl, "wb") as f:
        pickle.dump(processor, f)

    print(f"✅ Saved processed CSV: {output_csv}")
    print(f"✅ Saved processor PKL: {output_pkl}")


# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser(description="Train-time JSON processor for CHIMERA Task 2.")
    ap.add_argument("--input", required=True, help="Path to a JSON file or a directory containing JSON files.")
    ap.add_argument("--output_dir", default=None, help="Directory to save CSV + PKL (default: same as input).")
    ap.add_argument("--keep_label", action="store_true", help="Keep BRS label in CSV (for training).")
    args = ap.parse_args()

    in_path = Path(args.input)
    if in_path.is_dir():
        json_paths = sorted([p for p in in_path.glob("*.json") if p.is_file()])
        if not json_paths:
            raise FileNotFoundError(f"No .json files found in: {in_path}")
    else:
        if in_path.suffix.lower() != ".json":
            raise ValueError("Input must be a .json file or a folder with .json files")
        json_paths = [in_path]

    out_dir = Path(args.output_dir) if args.output_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    process_and_save(json_paths, out_dir, keep_label=args.keep_label)


if __name__ == "__main__":
    main()
