import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE  # NEW

# ================================
# 1. Config
# ================================
MICRO_ROOT = "MICRO"
PROCESSED_DIR = "processed_MICRO"
RANDOM_STATE = 42
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

BATCH_SIZE = 32
NUM_EPOCHS = 100
LR = 1e-4
NUM_WORKERS = 4
EARLY_STOPPING_PATIENCE = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================================
# 2. Data Transforms
# ================================
train_transform = A.Compose([
    A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Rotate(limit=25, p=0.6),
    A.RandomBrightnessContrast(p=0.4),
    A.GaussNoise(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ================================
# 3. Dataset
# ================================
class MicroParticleDataset(Dataset):
    def __init__(self, samples, transform=None):
        """
        samples: list of (path, label)
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)['image']
        return img, label, path

def collect_samples(root_dir):
    root_dir = Path(root_dir)
    classes = sorted([d.name for d in root_dir.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    samples = [
        (str(p), class_to_idx[cls])
        for cls in classes
        for p in (root_dir / cls).glob("*")
        if p.suffix.lower() in IMAGE_EXTS
    ]
    return samples, classes

# Collect training samples for SMOTE
train_samples, classes = collect_samples(Path(PROCESSED_DIR) / "train")
val_samples, _ = collect_samples(Path(PROCESSED_DIR) / "val")
test_samples, _ = collect_samples(Path(PROCESSED_DIR) / "test")
num_classes = len(classes)
print(f"Number of classes: {num_classes}, Classes: {classes}")

# ================================
# 4. SMOTE Oversampling
# ================================
# Extract features for SMOTE using flattened image histograms (lightweight proxy)
print("Applying SMOTE to balance training data...")

X, y = [], []
for path, label in tqdm(train_samples, desc="Preparing SMOTE features"):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64, 64))  # downsample for speed
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256]).flatten()
    X.append(hist)
    y.append(label)

X, y = np.array(X), np.array(y)

# Apply SMOTE
sm = SMOTE(random_state=RANDOM_STATE)
X_res, y_res = sm.fit_resample(X, y)

# Build oversampled training sample list
from collections import defaultdict
label_to_paths = defaultdict(list)
for p, label in train_samples:
    label_to_paths[label].append(p)

aug_train_samples = []
for label in np.unique(y_res):
    count = np.sum(y_res == label)
    available = label_to_paths[label]
    aug_train_samples.extend([
        (np.random.choice(available), label) for _ in range(count)
    ])

print(f"Original samples: {len(train_samples)}, After SMOTE: {len(aug_train_samples)}")

# ================================
# 5. Dataloaders
# ================================
train_ds = MicroParticleDataset(aug_train_samples, transform=train_transform)
val_ds = MicroParticleDataset(val_samples, transform=val_transform)
test_ds = MicroParticleDataset(test_samples, transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

# ================================
# 6. Model (ResNet50)
# ================================
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False  # freeze backbone

# Replace classifier (fc layer)
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, num_classes)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# ================================
# 7. Training Loop with Early Stopping
# ================================
best_val_loss = np.inf
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss, running_corrects, total = 0, 0, 0

    for imgs, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        running_corrects += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = running_corrects / total

    # Validation
    model.eval()
    val_loss, val_corrects, val_total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels, _ in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            val_corrects += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= val_total
    val_acc = val_corrects / val_total

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    scheduler.step(val_loss)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "resnet50_best.pth")
        #print("  Saved best model.")
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

# ================================
# 8. Test Evaluation
# ================================
model.load_state_dict(torch.load("resnet50_best.pth"))
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels, _ in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("Test Classification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))
