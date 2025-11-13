import joblib
import numpy as np
import cv2
from pathlib import Path
from sklearn.metrics import confusion_matrix
from skimage import io, color, feature
from scipy.stats import skew, kurtosis, entropy

# ===============================
# CONFIG
# ===============================
PROCESSED_DIR = "processed_MICRO"
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

# Load models + encoders
rf = joblib.load("rf.joblib")
svm = joblib.load("svm.joblib")
le = joblib.load("label_encoder.joblib")
scaler = joblib.load("scaler.joblib")

# ===============================
# FEATURE EXTRACTOR
# ===============================
def extract_classical_features(img_path):
    """Extract classical features for an image (same as training)."""
    try:
        img = io.imread(img_path)
    except:
        img = cv2.imread(img_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = color.rgb2gray(img)
    gray = cv2.resize((gray * 255).astype("uint8"), (64, 64))
    feats = []

    # Intensity
    feats += [
        np.mean(gray),
        np.std(gray),
        np.median(gray),
        np.var(gray),
        skew(gray.flatten()),
        kurtosis(gray.flatten()),
    ]

    # Texture (GLCM)
    glcm = feature.graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
    feats += [
        feature.graycoprops(glcm, "contrast")[0, 0],
        feature.graycoprops(glcm, "homogeneity")[0, 0],
        feature.graycoprops(glcm, "correlation")[0, 0],
    ]

    # Edge/contour
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(edges > 0)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)
    avg_contour_area = np.mean([cv2.contourArea(c) for c in contours]) if contours else 0
    feats += [edge_density, contour_count, avg_contour_area]

    # Entropy
    hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    feats.append(entropy(hist + 1e-7))

    return np.nan_to_num(np.array(feats, dtype=float))

# ===============================
# LOAD VALIDATION DATA
# ===============================
def load_val_data(processed_dir):
    val_dir = Path(processed_dir) / "val"
    X, y = [], []
    for cls_dir in val_dir.iterdir():
        if not cls_dir.is_dir():
            continue
        for p in cls_dir.glob("*"):
            if p.suffix.lower() in IMAGE_EXTS:
                feats = extract_classical_features(str(p))
                if feats is not None:
                    X.append(feats)
                    y.append(cls_dir.name)
    return np.array(X), np.array(y)

X_val, y_val = load_val_data(PROCESSED_DIR)
y_val_enc = le.transform(y_val)
X_val_scaled = scaler.transform(X_val)

# ===============================
# COMPUTE IoU
# ===============================
def compute_iou(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    ious = {}
    for i, cls in enumerate(classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        iou = TP / (TP + FP + FN + 1e-8)
        ious[cls] = iou
    return ious

# Random Forest
y_pred_rf = rf.predict(X_val_scaled)
ious_rf = compute_iou(y_val_enc, y_pred_rf, le.classes_)
print("\nRandom Forest IoU per class:")
for cls, iou in ious_rf.items():
    print(f"{cls}: {iou:.4f}")
print(f"Average IoU: {np.mean(list(ious_rf.values())):.4f}")

# SVM
y_pred_svm = svm.predict(X_val_scaled)
ious_svm = compute_iou(y_val_enc, y_pred_svm, le.classes_)
print("\nSVM IoU per class:")
for cls, iou in ious_svm.items():
    print(f"{cls}: {iou:.4f}")
print(f"Average IoU: {np.mean(list(ious_svm.values())):.4f}")
