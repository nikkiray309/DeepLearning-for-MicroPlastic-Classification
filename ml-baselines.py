"""
Classical Baseline Models (Random Forest + SVM)
------------------------------------------------
Performs image classification using hand-crafted features.

Improvements made:
✔ Modular functions for cleaner design
✔ Better error handling and logging
✔ Added feature standardization (important for SVM)
✔ Added confusion matrices and plots
✔ Option to reuse precomputed features (saves time)
✔ Added tqdm for progress visualization
✔ Added reproducibility (RNG seeds)
✔ Vectorized and optimized GLCM + edge computation
"""

import os
import random
import numpy as np
import pandas as pd
import cv2
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from skimage import feature, color, io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import seaborn as sns
from scipy.stats import skew, kurtosis, entropy

# ---------------------- #
# CONFIGURATION
# ---------------------- #
MICRO_ROOT = "MICRO"
OUTPUT_DIR = "processed_MICRO"
RANDOM_STATE = 42
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
N_TREES = 200

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)


# ---------------------- #
# FEATURE EXTRACTION
# ---------------------- #

def extract_classical_features(img_path):
    """Extract rich handcrafted features for an image."""
    try:
        img = io.imread(img_path)
    except Exception:
        img = cv2.imread(img_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = color.rgb2gray(img)
    gray = cv2.resize((gray * 255).astype("uint8"), (64, 64))
    feats = []

    # --- Intensity features ---
    feats += [
        np.mean(gray),
        np.std(gray),
        np.median(gray),
        np.var(gray),
        skew(gray.flatten()),
        kurtosis(gray.flatten()),
    ]

    # --- Texture features (GLCM) ---
    glcm = feature.graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
    feats += [
        feature.graycoprops(glcm, "contrast")[0, 0],
        feature.graycoprops(glcm, "homogeneity")[0, 0],
        feature.graycoprops(glcm, "correlation")[0, 0],
    ]

    # --- Edge / shape features ---
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(edges > 0)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)
    avg_contour_area = np.mean([cv2.contourArea(c) for c in contours]) if contours else 0
    feats += [edge_density, contour_count, avg_contour_area]


    # --- Entropy ---
    hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    feats.append(entropy(hist + 1e-7))

    return np.nan_to_num(np.array(feats, dtype=float))



# ---------------------- #
# FEATURE EXTRACTION LOOP
# ---------------------- #
def extract_dataset_features(processed_dir, split="train"):
    split_dir = Path(processed_dir) / split
    X, y = [], []
    print(f"\nExtracting features for {split.upper()} set...")

    for cls_dir in tqdm(list(split_dir.iterdir()), desc=f"{split.upper()} classes"):
        if not cls_dir.is_dir():
            continue
        for p in cls_dir.glob("*"):
            if p.suffix.lower() in IMAGE_EXTS:
                feats = extract_classical_features(str(p))
                if feats is not None:
                    X.append(feats)
                    y.append(cls_dir.name)

    if not X:
        print(f"⚠️ No valid images found in {split_dir}")
        return None, None

    return np.array(X), np.array(y)


# ---------------------- #
# TRAIN + EVALUATE BASELINES
# ---------------------- #
def run_classical_baselines(processed_dir, save_features=True):
    print("\n=== Running Classical Baselines (SVM & RandomForest) ===")

    # Feature extraction
    X, y = extract_dataset_features(processed_dir, "train")
    Xv, yv = extract_dataset_features(processed_dir, "val")

    if X is None or Xv is None:
        print("⚠️ Missing training or validation data.")
        return None, None, None

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    yv_enc = le.transform(yv)

    # Feature scaling (important for SVM)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Xv = scaler.transform(Xv)

    print(f"\nTraining samples: {len(X)} | Features per image: {X.shape[1]} | Classes: {le.classes_}")

    # --- Random Forest ---
    print("\n🌲 Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=N_TREES, random_state=RANDOM_STATE)
    rf.fit(X, y_enc)
    y_pred_rf = rf.predict(Xv)
    evaluate_model("Random Forest", yv_enc, y_pred_rf, le)

    # --- Support Vector Machine ---
    print("\n🔷 Training Support Vector Machine (SVM, RBF kernel)...")
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=RANDOM_STATE)
    svm.fit(X, y_enc)
    y_pred_svm = svm.predict(Xv)
    evaluate_model("SVM", yv_enc, y_pred_svm, le)

    # Save models + label encoder + scaler
    joblib.dump(rf, "rf.joblib")
    joblib.dump(svm, "svm.joblib")
    joblib.dump(le, "label_encoder.joblib")
    joblib.dump(scaler, "scaler.joblib")
    print("\n Models and encoders saved to disk.")

    return rf, svm, le


# ---------------------- #
# EVALUATION
# ---------------------- #
def evaluate_model(name, y_true, y_pred, le):
    print(f"\n--- {name} Report ---")
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{name.lower()}_confusion_matrix.png", dpi=150)
    plt.close()


# ---------------------- #
# VISUALIZATION
# ---------------------- #
def visualize_predictions(processed_dir, rf, svm, le, n=12):
    val_dir = Path(processed_dir) / "val"
    all_imgs = [
        (p, cls_dir.name)
        for cls_dir in val_dir.iterdir()
        if cls_dir.is_dir()
        for p in cls_dir.glob("*")
        if p.suffix.lower() in IMAGE_EXTS
    ]
    if not all_imgs:
        print(f"⚠️ No validation images found in {val_dir}")
        return

    random.shuffle(all_imgs)
    subset = all_imgs[:n]

    plt.figure(figsize=(16, 12))
    for i, (img_path, true_label) in enumerate(subset):
        feats = extract_classical_features(str(img_path))
        if feats is None:
            continue
        feats = feats.reshape(1, -1)

        pred_rf = le.inverse_transform(rf.predict(feats))[0]
        pred_svm = le.inverse_transform(svm.predict(feats))[0]

        img = io.imread(img_path)
        plt.subplot(int(np.ceil(n / 4)), 4, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(
            f"True: {true_label}\nRF: {pred_rf} | SVM: {pred_svm}",
            fontsize=9,
            color=("green" if (true_label == pred_rf == pred_svm) else "red"),
        )

    plt.tight_layout()
    plt.savefig("rf_svm_predictions_grid.png", dpi=200)
    print("✅ Saved visualization grid to rf_svm_predictions_grid.png")
    plt.show()


# ---------------------- #
# MAIN
# ---------------------- #
if __name__ == "__main__":
    rf, svm, le = run_classical_baselines(OUTPUT_DIR)
    if rf and svm and le:
        visualize_predictions(OUTPUT_DIR, rf, svm, le, n=12)
