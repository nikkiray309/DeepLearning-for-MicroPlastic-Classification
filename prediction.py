import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
import cv2
from torch.utils.data import Dataset
import os
# ================================
# 1. Config
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

PROCESSED_DIR = "processed_MICRO"
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

# ================================
# 2. Dataset
# ================================
class MicroParticleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = [
            (str(p), self.class_to_idx[cls])
            for cls in self.classes
            for p in (self.root_dir / cls).glob("*")
            if p.suffix.lower() in IMAGE_EXTS
        ]
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
    

test_ds = MicroParticleDataset(Path(PROCESSED_DIR)/"test", transform=None)  # no augmentations for visualization

# Get class names
classes = sorted([d.name for d in Path(PROCESSED_DIR).joinpath("train").iterdir() if d.is_dir()])

# ================================
# 3. Load Model
# ================================
from torchvision import models
import torch.nn as nn

num_classes = len(classes)
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
for param in model.features.parameters():
    param.requires_grad = False

model.classifier[6] = nn.Linear(4096, num_classes)
model.load_state_dict(torch.load("vgg16_best.pth", map_location=device))
model = model.to(device)
model.eval()

# ================================
# 4. Visualization function
# ================================
def show_and_save_predictions_grid(dataset, model, num_samples=12, save_path="vgg_predictions.png"):
    model.eval()
    indices = random.sample(range(len(dataset)), num_samples)
    
    cols = 4
    rows = (num_samples + cols - 1) // cols  # auto rows
    plt.figure(figsize=(cols*4, rows*4))
    
    for i, idx in enumerate(indices):
        img, label, path = dataset[idx]
        
        # Resize to 224x224 (VGG16 requirement)
        img = cv2.resize(img, (224, 224))
        
        # Convert to tensor and normalize like training
        img_tensor = torch.tensor(img / 255., dtype=torch.float32).permute(2,0,1)
        img_tensor = (img_tensor - torch.tensor([0.485,0.456,0.406]).view(3,1,1)) / torch.tensor([0.229,0.224,0.225]).view(3,1,1)
        input_img = img_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_img)
            pred = output.argmax(dim=1).item()
        
        # Convert back to numpy for plotting
        img_np = img_tensor.permute(1,2,0).cpu().numpy()
        img_np = np.clip(img_np * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406]), 0, 1)
        
        plt.subplot(rows, cols, i+1)
        plt.imshow(img_np)
        plt.axis('off')
        plt.title(f"True: {classes[label]}\nPred: {classes[pred]}")
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Predictions grid saved at: {save_path}")

# ================================
# 5. Run
# ================================
show_and_save_predictions_grid(test_ds, model, num_samples=12, save_path="vgg_predictions.png")
