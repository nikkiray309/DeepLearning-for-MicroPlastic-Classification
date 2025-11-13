#!/usr/bin/env python3
"""
ViT training on processed_MICRO with SMOTE, full evaluation, and Grad-CAM-like token heatmaps.
Saves logs + PNGs into SAVE_DIR.
"""

import os
from pathlib import Path
import random
import numpy as np
from collections import defaultdict
from PIL import Image
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision import models

from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# -------------------------
# 1. Config / Hyperparams
# -------------------------
DATA_DIR = Path("processed_MICRO")
SAVE_DIR = Path("vit_smote_results")
SAVE_DIR.mkdir(exist_ok=True, parents=True)

SEED = 42
BATCH_SIZE = 16
NUM_EPOCHS = 30
LR = 1e-4
EARLY_STOPPING_PATIENCE = 8
IMAGE_SIZE = 224
NUM_WORKERS = 4
SMOTE_RANDOM_STATE = 42
NUM_GRADCAM_SAMPLES = 6  # how many Grad-CAM images to save

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

# -------------------------
# 2. Helpers: dataset listing & SMOTE feature extraction
# -------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def collect_samples(split_dir: Path):
    """
    Collect (path, label_idx) pairs from a split directory structured as split/class/*.jpg
    Returns: samples list and classes list (sorted)
    """
    split_dir = Path(split_dir)
    classes = sorted([d.name for d in split_dir.parent.joinpath("train").iterdir() if d.is_dir()]) \
              if split_dir.name in {"val", "test"} and split_dir.parent.joinpath("train").exists() \
              else sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    # For robustness, if split_dir points to 'processed_MICRO/train' or similar
    if split_dir.exists() and any(split_dir.iterdir()):
        # if the directory itself has class subdirs
        classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])

    class_to_idx = {c: i for i, c in enumerate(classes)}
    samples = []
    for c in classes:
        cls_dir = split_dir / c
        if not cls_dir.exists():
            continue
        for p in cls_dir.glob("*"):
            if p.suffix in IMAGE_EXTS:
                samples.append((str(p), class_to_idx[c]))
    return samples, classes


def rgb_hist_feature(image_path, size=(64, 64), bins=(8, 8, 8)):
    """
    Read image, resize to `size`, compute 3D RGB histogram and flatten to feature vector.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(size, Image.BILINEAR)
    a = np.asarray(img)
    # compute 3D hist
    hist, _ = np.histogramdd(
        a.reshape(-1, 3),
        bins=bins,
        range=[(0, 256), (0, 256), (0, 256)]
    )
    hist = hist.flatten().astype(np.float32)
    # normalize
    hist = hist / (hist.sum() + 1e-9)
    return hist


# -------------------------
# 3. Build oversampled training samples using SMOTE
# -------------------------
train_samples, classes = collect_samples(DATA_DIR / "train")
val_samples, _ = collect_samples(DATA_DIR / "val")
test_samples, _ = collect_samples(DATA_DIR / "test")

if len(train_samples) == 0:
    raise RuntimeError(f"No training images found under {DATA_DIR/'train'}. Check your folder structure.")

num_classes = len(classes)
print(f"Found classes ({num_classes}): {classes}")
print(f"Train samples: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

# Build features X, labels y for SMOTE
print("Extracting lightweight features for SMOTE (RGB histograms)...")
X = []
y = []
for path, label in tqdm(train_samples, desc="SMOTE features"):
    X.append(rgb_hist_feature(path, size=(64, 64)))
    y.append(label)
X = np.stack(X, axis=0)
y = np.array(y)

# Apply SMOTE
print("Applying SMOTE...")
sm = SMOTE(random_state=SMOTE_RANDOM_STATE)
X_res, y_res = sm.fit_resample(X, y)
print(f"SMOTE: {len(y)} -> {len(y_res)} samples")

# Map original label -> available paths
label_to_paths = defaultdict(list)
for p, l in train_samples:
    label_to_paths[l].append(p)

# Build augmented train sample list by sampling image paths according to y_res labels.
print("Building oversampled training list (mapping synthetic samples to real image paths by class sampling)...")
aug_train_samples = []
for lbl in y_res:
    choices = label_to_paths.get(lbl)
    if not choices:
        # fallback: randomly pick from all train images
        choices = [p for p, _ in train_samples]
    aug_train_samples.append((str(random.choice(choices)), int(lbl)))

print(f"Augmented train samples: {len(aug_train_samples)}")

# -------------------------
# 4. PyTorch Dataset that accepts a sample list (path,label)
# -------------------------
class PathDataset(Dataset):
    def __init__(self, samples, transform=None):
        """
        samples: list of (path_str, label_int)
        transform: torchvision transforms applied to PIL image
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, path


# -------------------------
# 5. Transforms, Dataloaders
# -------------------------
train_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(20),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

val_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

train_ds = PathDataset(aug_train_samples, transform=train_transform)
val_ds = PathDataset(val_samples, transform=val_transform)
test_ds = PathDataset(test_samples, transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

# -------------------------
# 6. Build ViT model (freeze backbone, train head)
# -------------------------
print("Initializing ViT model (fine-tuning head)...")
vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
# Freeze encoder
for param in vit.encoder.parameters():
    param.requires_grad = False

# replace head
in_features = vit.heads.head.in_features
vit.heads.head = nn.Linear(in_features, num_classes)
vit = vit.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vit.heads.head.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

# -------------------------
# 7. Training loop (log + save)
# -------------------------
best_val_loss = float("inf")
patience_counter = 0

train_losses = []
val_losses = []
train_accs = []
val_accs = []

log_path = SAVE_DIR / "train_log.txt"
if log_path.exists():
    log_path.unlink()  # overwrite

print("Starting training...")
for epoch in range(1, NUM_EPOCHS + 1):
    vit.train()
    running_loss = 0.0
    running_corrects = 0
    running_total = 0

    for imgs, labels, _ in tqdm(train_loader, desc=f"Train Epoch {epoch}/{NUM_EPOCHS}"):
        imgs = imgs.to(DEVICE)
        labels = torch.tensor(labels, dtype=torch.long, device=DEVICE)

        optimizer.zero_grad()
        outputs = vit(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        running_corrects += (preds == labels).sum().item()
        running_total += labels.size(0)

    epoch_train_loss = running_loss / running_total
    epoch_train_acc = running_corrects / running_total
    train_losses.append(epoch_train_loss)
    train_accs.append(epoch_train_acc)

    # Validation
    vit.eval()
    val_running_loss = 0.0
    val_running_corrects = 0
    val_running_total = 0
    with torch.no_grad():
        for imgs, labels, _ in val_loader:
            imgs = imgs.to(DEVICE)
            labels = torch.tensor(labels, dtype=torch.long, device=DEVICE)
            outputs = vit(imgs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            val_running_corrects += (preds == labels).sum().item()
            val_running_total += labels.size(0)

    epoch_val_loss = val_running_loss / max(1, val_running_total)
    epoch_val_acc = val_running_corrects / max(1, val_running_total)
    val_losses.append(epoch_val_loss)
    val_accs.append(epoch_val_acc)

    scheduler.step(epoch_val_loss)

    log_line = f"Epoch {epoch}: Train Loss {epoch_train_loss:.4f} Acc {epoch_train_acc:.4f} | Val Loss {epoch_val_loss:.4f} Acc {epoch_val_acc:.4f}"
    print(log_line)
    with open(log_path, "a") as f:
        f.write(log_line + "\n")

    # Save best
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        patience_counter = 0
        torch.save(vit.state_dict(), SAVE_DIR / "vit_smote_best.pth")
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

# -------------------------
# 8. Save loss curves
# -------------------------
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="train_loss")
plt.plot(val_losses, label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ViT train/val loss (with SMOTE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(SAVE_DIR / "vit_loss_curves.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(train_accs, label="train_acc")
plt.plot(val_accs, label="val_acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ViT train/val accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(SAVE_DIR / "vit_acc_curves.png")
plt.close()

# -------------------------
# 9. Test evaluation: predictions, classification report, confusion matrix, IoU
# -------------------------
print("Evaluating on test set...")
vit.load_state_dict(torch.load(SAVE_DIR / "vit_smote_best.pth", map_location=DEVICE))
vit.eval()

all_preds = []
all_labels_list = []
all_paths = []

with torch.no_grad():
    for img, label, path in tqdm(test_ds, desc="Iterate test samples"):
        # test_ds returns (img_pil_transformed, label, path)
        # our test_ds is PathDataset so __getitem__ returns (tensor, label, path)
        if isinstance(img, torch.Tensor):
            inp = img.unsqueeze(0).to(DEVICE)
        else:
            # transform then unsqueeze
            inp = val_transform(img).unsqueeze(0).to(DEVICE)
        out = vit(inp)
        pred = out.argmax(dim=1).item()
        all_preds.append(pred)
        all_labels_list.append(int(label))
        all_paths.append(path)

# classification report
report = classification_report(all_labels_list, all_preds, target_names=classes, digits=4)
print(report)
(SAVE_DIR / "classification_report.txt").write_text(report)

# confusion matrix
cm = confusion_matrix(all_labels_list, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("ViT Confusion Matrix (SMOTE)")
plt.tight_layout()
plt.savefig(SAVE_DIR / "vit_confusion_matrix.png")
plt.close()

# IoU per class (derived from confusion matrix)
# IoU = TP / (TP + FP + FN)
cm = cm.astype(np.int64)
tp = np.diag(cm)
fp = cm.sum(axis=0) - tp
fn = cm.sum(axis=1) - tp
iou = tp / (tp + fp + fn + 1e-9)
with open(SAVE_DIR / "iou_per_class.txt", "w") as f:
    for cls_name, cls_iou in zip(classes, iou):
        f.write(f"{cls_name}: {cls_iou:.4f}\n")
print("Saved IoU per class.")

# -------------------------
# 10. Sample predictions grid (save PNG)
# -------------------------
def save_sample_predictions_grid(paths, true_labels, pred_labels, classes, save_path, n=12):
    n = min(n, len(paths))
    indices = random.sample(range(len(paths)), n)
    cols = 4
    rows = math.ceil(n / cols)
    plt.figure(figsize=(cols * 3, rows * 3))
    for i, idx in enumerate(indices):
        img = Image.open(paths[idx]).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(f"T: {classes[true_labels[idx]]}\nP: {classes[pred_labels[idx]]}", fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

save_sample_predictions_grid(all_paths, all_labels_list, all_preds, classes, SAVE_DIR / "vit_sample_predictions.png")
print("Saved sample predictions grid.")

# -------------------------
# 11. Grad-CAM like visualization for ViT (token-based)
# -------------------------
# We'll hook last encoder layer output (tokens) and its gradient, aggregate per-token importance, reshape to patch grid.
class ViTTokenGradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        self.hook_handles = []
        # hook on the last encoder layer output (before final layernorm) - using encoder.layers[-1].mlp or .ln_2 etc.
        # We'll hook on the output of encoder.layers[-1] by registering a forward hook on that layer object.
        target_module = self.model.encoder.layers[-1]
        # forward hook to capture activations (the module returns shaped (B, N, C))
        def forward_hook(module, inp, out):
            # out is probably the Transformer output for that sublayer; capturing the full tokens ideally:
            # to be safe we'll capture module.self_attn._get_attn_output or out if shaped tokens.
            self.activations = out  # tensor shape (B, N, C) typical
        def backward_hook(module, grad_in, grad_out):
            # grad_out[0] corresponds to gradient w.r.t. out
            self.gradients = grad_out[0]
        self.hook_handles.append(target_module.register_forward_hook(forward_hook))
        # backward hook (use full backward hook if available) — fallback to register_backward_hook
        try:
            self.hook_handles.append(target_module.register_full_backward_hook(backward_hook))
        except Exception:
            self.hook_handles.append(target_module.register_backward_hook(backward_hook))

    def clear(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles = []

    def generate_cam(self, input_tensor, class_idx=None):
        """
        input_tensor: 1xC x H x W tensor (already normalized and on DEVICE)
        returns: CAM upsampled to input HxW as numpy [0..1]
        """
        self.activations = None
        self.gradients = None

        # forward
        out = self.model(input_tensor)  # shape (1, num_classes)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()

        # backward on selected class score
        self.model.zero_grad()
        score = out[0, class_idx]
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Activations or gradients not captured. Hooks failed.")

        # activations: (1, N, C), gradients: (1, N, C)
        act = self.activations.detach()  # (1, N, C)
        grad = self.gradients.detach()  # (1, N, C)
        # compute token weights by global-average grad over channels
        weights = grad.mean(dim=2)  # (1, N)
        # weighted sum of activations: sum_{tokens} weights[token] * activation[token].mean_across_channels
        # We'll compute token importance: token_score = (weights * act.mean(dim=2)).squeeze()
        token_scores = (weights.squeeze(0) * act.mean(dim=2).squeeze(0))  # (N,)
        # drop class token (first token) if present — vit has class token at position 0
        if token_scores.size(0) > 1:
            token_scores = token_scores[1:]  # shape (N-1,)
        # reshape tokens to patch grid
        # compute grid size: sqrt(N-1)
        n_tokens = token_scores.numel()
        grid_size = int(math.sqrt(n_tokens))
        if grid_size * grid_size != n_tokens:
            # fallback: try nearest square by truncation or padding
            grid_size = int(math.floor(math.sqrt(n_tokens)))
            n_use = grid_size * grid_size
            token_scores = token_scores[:n_use]
        cam = token_scores.reshape(grid_size, grid_size).cpu().numpy()
        # normalize CAM
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / (cam.max() + 1e-9)
        # upsample to image size
        cam_img = Image.fromarray(np.uint8(cam * 255)).resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR)
        cam_np = np.asarray(cam_img).astype(np.float32) / 255.0
        return cam_np

# generate Grad-CAM comparisons for a few test images
gradcam = ViTTokenGradCAM(vit)
for i in range(min(NUM_GRADCAM_SAMPLES, len(all_paths))):
    path = all_paths[i]
    img_pil = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    # prepare tensor using val_transform (we used torchvision transforms)
    inp = val_transform(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = vit(inp)
        pred = out.argmax(dim=1).item()
    # need grads, so run generate_cam (it will run forward + backward internally)
    cam = gradcam.generate_cam(inp, class_idx=pred)  # HxW float [0..1]
    # make heatmap
    heatmap = plt.cm.jet(cam)[:, :, :3]  # HxWx3
    overlay = (np.array(img_pil).astype(np.float32) / 255.0 * 0.5 + heatmap * 0.5)
    overlay = np.clip(overlay, 0, 1)
    # combine side-by-side (orig | heatmap overlay)
    combined = np.concatenate([np.array(img_pil) / 255.0, overlay], axis=1)
    plt.figure(figsize=(8, 4))
    plt.imshow(combined)
    plt.title(f"True: {classes[all_labels_list[i]]} | Pred: {classes[pred]}")
    plt.axis("off")
    savep = SAVE_DIR / f"gradcam_comparison_{i+1}.png"
    plt.tight_layout()
    plt.savefig(savep, dpi=150)
    plt.close()
    print("Saved", savep)

# cleanup hooks
gradcam.clear()

print("All outputs saved to:", SAVE_DIR.resolve())