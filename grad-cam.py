import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
from torch.utils.data import Dataset

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

test_ds = MicroParticleDataset(Path(PROCESSED_DIR)/"test", transform=None)
classes = test_ds.classes
# ================================
# 3. Load VGG16 Model  / Resnet-50 Model based on the requirement
# ================================
from torchvision import models
num_classes = len(test_ds.classes)
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
for param in model.features.parameters():
    param.requires_grad = False
model.classifier[6] = nn.Linear(4096, num_classes)
model.load_state_dict(torch.load("vgg16_best.pth", map_location=device))
model = model.to(device)
model.eval()

# ================================
# 4. Grad-CAM Class
# ================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        self.target_layer = target_layer
        self._register_hooks()

    def _register_hooks(self):
        
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        # Ensure the input tensor requires gradients if it doesn't already
        if not input_tensor.requires_grad:
            input_tensor.requires_grad_(True)
            
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        # Select the score for the predicted class
        score = output[0, class_idx]
        
        # Backpropagate through the selected score
        score.backward(retain_graph=True)

        if self.gradients is None:
            raise RuntimeError("Gradients not captured. Check hook registration!")

        
        weights = self.gradients.mean(dim=(2,3), keepdim=True)
        # We need the activations to be detached for the final calculation
        cam = (weights * self.activations.detach()).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

target_layer = model.features[-1]
gradcam = GradCAM(model, target_layer)

# ================================
# 5. Generate and Save Grad-CAM Images
# ================================
num_samples = 6
indices = random.sample(range(len(test_ds)), num_samples)

for i, idx in enumerate(indices):
    img, label, path = test_ds[idx]
    img_resized = cv2.resize(img, (224, 224))
    img_tensor = torch.tensor(img_resized/255., dtype=torch.float32).permute(2,0,1)
    img_tensor = (img_tensor - torch.tensor([0.485,0.456,0.406]).view(3,1,1)) / torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    input_tensor = img_tensor.unsqueeze(0).to(device)

    # Prediction
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()

    # Grad-CAM
    cam = gradcam.generate(input_tensor, class_idx=pred_class)
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = 0.5*img_resized + 0.5*heatmap
    overlay = np.clip(overlay,0,255).astype(np.uint8)

    # Combine original and Grad-CAM
    combined_width = img_resized.shape[1]*2
    combined = np.zeros((img_resized.shape[0], combined_width, 3), dtype=np.uint8)
    combined[:, :img_resized.shape[1], :] = img_resized
    combined[:, img_resized.shape[1]:, :] = overlay

    # Plot with labels
    plt.figure(figsize=(8,4))
    plt.imshow(combined)
    plt.axis('off')
    plt.title(f"True: {classes[label]} | Pred: {classes[pred_class]}", fontsize=12)
    save_path = f"comparison_gradcam_labeled_{i+1}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved labeled side-by-side comparison: {save_path}")
