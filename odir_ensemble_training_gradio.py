# ODIR-2019 Multi-Class Ensemble Training + Gradio App
# This notebook trains EfficientNet-B4 + YOLO-CLS ensemble and exposes a Gradio app for inference.

# 1️⃣ Install required packages
!pip install ultralytics timm albumentations wandb gradio scikit-learn

# 2️⃣ Imports
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score, confusion_matrix
import wandb
from ultralytics import YOLO
import gradio as gr

# 3️⃣ Dataset setup
data_dir = Path('ODIR-2019/YOLO/processed_512g_merged')
train_root = data_dir / 'train'
val_root = data_dir / 'val'
test_root = data_dir / 'test'

CLASSES = sorted([d.name for d in train_root.iterdir() if d.is_dir()])
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c:i for i,c in enumerate(CLASSES)}

# Compute class weights from folder counts
counts = torch.tensor([len(list((train_root / cls).glob('*'))) for cls in CLASSES], dtype=torch.float32)
probs = counts / counts.sum()
weights = (probs.max() / probs)

# 4️⃣ Albumentations transforms
train_tfms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5),
    A.RandomResizedCrop(380,380, scale=(0.9,1.0), ratio=(0.95,1.05)),
    A.RandomBrightnessContrast(0.1,0.1,p=0.5),
    A.RandomGamma(gamma_limit=(90,110), p=0.3),
    A.Normalize(),
    ToTensorV2()
])

val_tfms = A.Compose([
    A.Resize(380,380),
    A.Normalize(),
    ToTensorV2()
])

# 5️⃣ Dataset class
class ODIRFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        for cls in CLASSES:
            for img_path in (self.root / cls).glob('*'):
                self.samples.append((str(img_path), CLASS_TO_IDX[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)['image']
        return img, label

# 6️⃣ DataLoaders
train_ds = ODIRFolderDataset(train_root, train_tfms)
val_ds = ODIRFolderDataset(val_root, val_tfms)
test_ds = ODIRFolderDataset(test_root, val_tfms)

train_loader = DataLoader(train_ds, batch_size=12, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=12, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=12, shuffle=False, num_workers=4, pin_memory=True)

# 7️⃣ EfficientNet-B4 model
class EffNetB4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=num_classes)
    def forward(self, x):
        return self.model(x)

model_eff = EffNetB4(NUM_CLASSES).cuda()
criterion = nn.CrossEntropyLoss(weight=weights.cuda())
optimizer = torch.optim.AdamW(model_eff.parameters(), lr=3e-4)

# 8️⃣ Training loop with best model saving
import copy
best_f1 = 0
save_path = 'best_effnet_b4.pth'
for epoch in range(25):
    model_eff.train()
    losses = []
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        logits = model_eff(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    avg_loss = np.mean(losses)

    # Validation
    model_eff.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.cuda()
            logits = model_eff(x)
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(y.numpy())
    macro_f1 = f1_score(labels, preds, average='macro')
    print(f'Epoch {epoch} | Loss {avg_loss:.4f} | Macro-F1 {macro_f1:.4f}')

    if macro_f1 > best_f1:
        best_f1 = macro_f1
        torch.save(model_eff.state_dict(), save_path)
        print(f'Saved best EfficientNet model at epoch {epoch}')

# 9️⃣ Load best EfficientNet for evaluation
torch.cuda.empty_cache()
model_eff.load_state_dict(torch.load(save_path))
model_eff.eval()

# 10️⃣ Load YOLO-CLS best checkpoint
yolo_model = YOLO('runs_odir/yolo_cls_macro_f1/weights/best.pt')
yolo_model.eval()

# 11️⃣ Helper functions for predictions

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_aug = val_tfms(image=img)['image'].unsqueeze(0).cuda()
    return img_aug, img

def eff_predict(img_tensor):
    with torch.no_grad():
        logits = model_eff(img_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs

def yolo_predict(img_path):
    r = yolo_model(img_path, verbose=False)[0]
    probs = r.probs.data.cpu().numpy()[0]
    return probs

def ensemble_predict(eff_probs, yolo_probs, alpha=0.7):
    return alpha*eff_probs + (1-alpha)*yolo_probs

# 12️⃣ Gradio app function
def predict(image_path):
    img_tensor, _ = preprocess_image(image_path)
    eff_probs = eff_predict(img_tensor)
    yolo_probs = yolo_predict(image_path)
    fused_probs = ensemble_predict(eff_probs, yolo_probs)
    pred_class = CLASSES[np.argmax(fused_probs)]
    result = {cls: float(fused_probs[i]) for i, cls in enumerate(CLASSES)}
    return pred_class, result

# 13️⃣ Launch Gradio app
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='filepath'),
    outputs=[gr.Textbox(label='Predicted Class'), gr.Label(num_top_classes=8, label='Class Probabilities')],
    title='ODIR-2019 Fundus Multi-Class Classifier',
    description='Ensemble of EfficientNet-B4 + YOLO-CLS'
)

iface.launch(share=True)
