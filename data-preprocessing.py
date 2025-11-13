import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import glob
import joblib

# Machine learning / deep learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from skimage import feature, color, io

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2

MICRO_ROOT = "MICRO"        
OUTPUT_DIR = "processed_MICRO"       # where processed crops and splits will be saved
RANDOM_STATE = 42
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

# Hyperparams
BATCH_SIZE = 32
NUM_EPOCHS = 20
LR = 1e-4
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# UTIL: safe read TSV
# -----------------------
def read_annotation_tsv(path):
    """Try reading TSV with pandas using common separators."""
    try:
        df = pd.read_csv(path, sep='\t', engine='python')
    except Exception:
        try:
            df = pd.read_csv(path, sep='\s+', engine='python', header=0)
        except Exception:
            df = pd.read_csv(path, engine='python')
    return df
# -----------------------
# STEP 1: Preprocess & Crop
# -----------------------
def preprocess_and_crop(micro_root, out_dir, rebuild=False):
    """
    Parse annotation TSVs and crop particle images into out_dir/all/<class>/
    Then create train/val/test splits in out_dir/{train,val,test}/<class>/
    """
    micro_root = Path(micro_root)
    out_dir = Path(out_dir)
    all_dir = out_dir / "all"
    if rebuild and out_dir.exists():
        print("Removing existing processed directory because rebuild=True...")
        shutil.rmtree(out_dir)
    all_dir.mkdir(parents=True, exist_ok=True)

    annotation_dir = micro_root / "annotation"
    raw_dir = micro_root / "raw_img"
    tsv_files = [p for p in annotation_dir.iterdir() if p.suffix.lower() == ".tsv"]
    print(f"Found {len(tsv_files)} tsv files")

    sample_rows = []
    crop_count = 0

    for tsv_path in tqdm(tsv_files, desc="Processing TSVs"):
        df = read_annotation_tsv(tsv_path)
        # Expect columns: ID minr minc maxr maxc class  (if header missing, try to infer)
        cols = [c.lower() for c in df.columns]
        # Normalize column names
        col_map = {}
        for col in df.columns:
            low = col.lower()
            if 'minr' in low: col_map[col] = 'minr'
            if 'minc' in low: col_map[col] = 'minc'
            if 'maxr' in low: col_map[col] = 'maxr'
            if 'maxc' in low: col_map[col] = 'maxc'
            if low == 'id' or 'id' == low: col_map[col] = 'ID'
            if 'class' in low: col_map[col] = 'class'
        df = df.rename(columns=col_map)

        # Determine raw image filename this TSV corresponds to:
        # TSV example: "08_03_micro_line_only1.JPG.tsv" -> raw image "08_03_micro_line_only1.JPG"
        raw_img_name = tsv_path.name
        if raw_img_name.endswith(".tsv"):
            raw_img_name = raw_img_name[:-4]
        raw_img_path = raw_dir / raw_img_name
        if not raw_img_path.exists():
            # try uppercase/lowercase
            for ext in IMAGE_EXTS:
                alt = raw_img_path.with_suffix(ext)
                if alt.exists():
                    raw_img_path = alt
                    break
        if not raw_img_path.exists():
            # skip if raw image not found
            # print(f"Warning: raw image not found for {tsv_path.name}, expected {raw_img_path}")
            continue

        img = cv2.imread(str(raw_img_path))
        if img is None:
            continue
        h,w = img.shape[:2]

        # iterate rows
        for i, row in df.iterrows():
            try:
                minr = int(row['minr'])
                minc = int(row['minc'])
                maxr = int(row['maxr'])
                maxc = int(row['maxc'])
            except Exception:
                continue
            # clamp coords
            minr = max(0, min(minr, h-1))
            minc = max(0, min(minc, w-1))
            maxr = max(0, min(maxr, h))
            maxc = max(0, min(maxc, w))
            if maxr <= minr or maxc <= minc:
                continue
            crop = img[minr:maxr, minc:maxc]
            if crop.size == 0:
                continue
            # class label may be like "line/2022-03-28 ..."
            lbl = str(row.get('class', '')).strip()
            if "/" in lbl:
                lbl = lbl.split("/")[0]
            if lbl == "" or lbl.lower() in ["nan", "none"]:
                lbl = "unknown"

            save_cls_dir = all_dir / lbl
            save_cls_dir.mkdir(parents=True, exist_ok=True)
            out_name = f"{tsv_path.stem}_{i}_{lbl}.jpg"
            out_path = save_cls_dir / out_name
            # convert BGR->RGB for saving visual consistency
            cv2.imwrite(str(out_path), crop)
            sample_rows.append({"img": str(out_path), "label": lbl})
            crop_count += 1

    print(f"Cropped {crop_count} samples into {all_dir}")

    # Create splits
    samples_df = pd.DataFrame(sample_rows)
    # drop duplicates
    samples_df = samples_df.drop_duplicates(subset=["img"]).reset_index(drop=True)
    # drop unknown label images if too many
    samples_df = samples_df[samples_df['label'] != 'unknown']

    print("Class distribution before split:")
    print(samples_df['label'].value_counts())

    train_df, testval_df = train_test_split(samples_df, test_size=0.30, stratify=samples_df['label'], random_state=RANDOM_STATE)
    val_df, test_df = train_test_split(testval_df, test_size=0.5, stratify=testval_df['label'], random_state=RANDOM_STATE)

    # Save splits to disk
    for split_name, df in zip(["train","val","test"], [train_df, val_df, test_df]):
        for _, r in df.iterrows():
            src = Path(r['img'])
            cls = r['label']
            dest_dir = out_dir / split_name / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dest_dir / src.name)

    print("Saved splits under:", out_dir)
    print("Train/Val/Test sizes:", len(train_df), len(val_df), len(test_df))
    return out_dir

# If run as main, execute preprocessing
if __name__ == "__main__":
    processed = preprocess_and_crop(MICRO_ROOT, OUTPUT_DIR, rebuild=False)
    print("Data preprocessing and cropping complete.")
    print("Processed data available at:", processed)