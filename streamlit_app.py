# ============================================================
# Ishihara Card Classification using Vision Transformer (ViT)
# ============================================================

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from transformers import ViTForImageClassification, ViTImageProcessor

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
class CFG:
    seed = 42
    num_epochs = 10
    batch_size = 16
    lr = 2e-5
    num_folds = 5
    num_labels = 3
    image_size = 224
    model_name = "google/vit-base-patch16-224-in21k"
    data_dir = "dataset"
    output_dir = "models"

os.makedirs(CFG.output_dir, exist_ok=True)

# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CFG.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
class IshiharaDataset(Dataset):
    def __init__(self, df, processor, transform=None):
        self.df = df
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["image_path"]
        label = int(self.df.iloc[idx]["label"])

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        return inputs, label

# ------------------------------------------------------------
# Load Data
# ------------------------------------------------------------
# Example expected format:
#   dataset/
#       normal/
#           img1.jpg
#           img2.jpg
#       red_green/
#           ...
#       total/
#           ...
def load_data(data_dir):
    data = []
    class_names = sorted(os.listdir(data_dir))
    for label, cls in enumerate(class_names):
        img_dir = os.path.join(data_dir, cls)
        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)
            data.append({"image_path": img_path, "label": label})
    return pd.DataFrame(data), class_names

df, class_names = load_data(CFG.data_dir)
print("Total images:", len(df))
print("Classes:", class_names)

# ------------------------------------------------------------
# Model Training
# ------------------------------------------------------------
processor = ViTImageProcessor.from_pretrained(CFG.model_name)

transform = transforms.Compose([
    transforms.Resize((CFG.image_size, CFG.image_size)),
    transforms.ToTensor()
])

skf = StratifiedKFold(n_splits=CFG.num_folds, shuffle=True, random_state=CFG.seed)

fold_accuracies = []
best_acc = 0.0
best_model_path = None

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
    print(f"\n========== Fold {fold + 1} / {CFG.num_folds} ==========")

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_dataset = IshiharaDataset(train_df, processor, transform)
    val_dataset = IshiharaDataset(val_df, processor, transform)

    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False)

    model = ViTForImageClassification.from_pretrained(
        CFG.model_name, num_labels=CFG.num_labels
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr)
    criterion = nn.CrossEntropyLoss()

    best_fold_acc = 0.0

    for epoch in range(CFG.num_epochs):
        model.train()
        train_loss = 0

        for batch_inputs, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG.num_epochs}"):
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            batch_labels = batch_labels.to(device)

            outputs = model(**batch_inputs)
            loss = criterion(outputs.logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch_inputs, batch_labels in val_loader:
                batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
                batch_labels = batch_labels.to(device)

                outputs = model(**batch_inputs)
                preds = torch.argmax(outputs.logits, dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(batch_labels.cpu().numpy())

        acc = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch+1}: Val Accuracy = {acc:.4f}")

        if acc > best_fold_acc:
            best_fold_acc = acc
            fold_model_path = os.path.join(CFG.output_dir, f"vit_fold{fold+1}.pth")
            torch.save(model.state_dict(), fold_model_path)

    fold_accuracies.append(best_fold_acc)
    print(f"Best accuracy for fold {fold+1}: {best_fold_acc:.4f}")

    # Save the best model overall
    if best_fold_acc > best_acc:
        best_acc = best_fold_acc
        best_model_path = fold_model_path

# ------------------------------------------------------------
# Evaluation Summary
# ------------------------------------------------------------
print("\n========== Cross-validation Summary ==========")
for i, acc in enumerate(fold_accuracies, 1):
    print(f"Fold {i}: {acc:.4f}")
print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f}")
print(f"Best model saved to: {best_model_path}")

# ------------------------------------------------------------
# Save final best model
# ------------------------------------------------------------
best_vit_model = ViTForImageClassification.from_pretrained(
    CFG.model_name, num_labels=CFG.num_labels
)
best_vit_model.load_state_dict(torch.load(best_model_path, map_location="cpu"))
torch.save(best_vit_model.state_dict(), os.path.join(CFG.output_dir, "best_vit_model.pth"))
print("Saved best_vit_model.pth successfully.")
