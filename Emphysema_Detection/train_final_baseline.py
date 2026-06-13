"""
train_final_baseline.py
=======================
Scientifically Valid Baseline for Emphysema Detection

This script trains three architectures (ResNet-50, DenseNet-121, EfficientNet-B0)
exclusively on the patient-stratified NIH internal dataset. It evaluates the 
models on an internal NIH test set, and optionally performs zero-shot external 
validation on ChestX6 (with the caveat that ChestX6 contains known acquisition 
confounds).

This script generates publication-ready training curves and ROC-AUC plots,
saving all outputs to the 'results' subfolder.
"""

import os
import re
import random
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from PIL import Image

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
BASE_DIR = r"C:\Users\batch_dcahfiw\Downloads\datasetfinalforcopdemphysema"
DATASET_DIR = os.path.join(BASE_DIR, "Final_Dataset_PatientSafe")
PROV_CSV = os.path.join(BASE_DIR, "provenance_report.csv")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 32
NUM_WORKERS = 10
MAX_EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

LMAP = {"Normal": 0, "Emphysema": 1}
IMAGENET_M = [0.485, 0.456, 0.406]
IMAGENET_S = [0.229, 0.224, 0.225]

# ---------------------------------------------------------
# Patient Splitting Logic
# ---------------------------------------------------------
def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def extract_patient_id(filename, dataset):
    if dataset == "ChestX6":
        stem = filename.rsplit(".", 1)[0]
        if "Emphysema" in stem:
            parts = stem.split("_", 1)
            return "E_" + parts[1] if len(parts) == 2 else "E_" + stem
        else:
            m = re.search(r"\((\d+)\)", stem)
            return "N_" + m.group(1) if m else "N_" + stem
    else:
        return filename.split("_")[0] if "_" in filename else filename

def create_patient_splits(df, seed=42):
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(gss1.split(df, groups=df["patient_id"]))
    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()
    
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["patient_id"]))
    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()
    return train_df, val_df, test_df

def create_stratified_external_test(df, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    parts_te = []
    for cls in ["Normal", "Emphysema"]:
        sub = df[df["class_label"] == cls].copy()
        if len(sub) == 0: continue
        if len(sub["patient_id"].unique()) < 2: continue
        _, te_i = next(gss.split(sub, groups=sub["patient_id"]))
        parts_te.append(sub.iloc[te_i])
    te = pd.concat(parts_te).sample(frac=1, random_state=seed).reset_index(drop=True)
    return te

# ---------------------------------------------------------
# Dataset & Model
# ---------------------------------------------------------
class XRayDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.labels = [LMAP[r] for r in self.df["class_label"]]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["abs_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]

def get_transforms():
    tf_tr = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_M, IMAGENET_S)
    ])
    tf_te = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_M, IMAGENET_S)
    ])
    return tf_tr, tf_te

class EmphysemaClassifier(nn.Module):
    def __init__(self, arch="resnet50"):
        super().__init__()
        self.arch = arch
        if arch == "resnet50":
            self.model = models.resnet50(weights="IMAGENET1K_V1")
            in_features = self.model.fc.in_features
            self.model.fc = self._build_head(in_features)
        elif arch == "densenet121":
            self.model = models.densenet121(weights="IMAGENET1K_V1")
            in_features = self.model.classifier.in_features
            self.model.classifier = self._build_head(in_features)
        elif arch == "efficientnet_b0":
            self.model = models.efficientnet_b0(weights="IMAGENET1K_V1")
            in_features = self.model.classifier[1].in_features
            self.model.classifier = self._build_head(in_features)
            
    def _build_head(self, in_features):
        return nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
        
    def forward(self, x):
        return self.model(x)

# ---------------------------------------------------------
# Execution
# ---------------------------------------------------------
def evaluate(model, loader, name):
    yt, yp, ypr = [], [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            out = model(imgs.to(DEVICE))
            yp.extend(out.argmax(dim=1).cpu().numpy())
            ypr.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
            yt.extend(lbls.numpy())
    acc = accuracy_score(yt, yp)
    prec = precision_score(yt, yp, zero_division=0)
    rec = recall_score(yt, yp, zero_division=0)
    f1 = f1_score(yt, yp, zero_division=0)
    auc = roc_auc_score(yt, ypr)
    cm = confusion_matrix(yt, yp)
    return yt, ypr, cm, {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc}

def main():
    import sys
    print("="*65)
    print("  FINAL MULTI-MODEL BASELINE TRAINING (SCIENTIFICALLY VALID)")
    print("="*65)
    set_seed(SEED)

    prov_df = pd.read_csv(PROV_CSV)
    
    nih_df = prov_df[prov_df["original_dataset"] == "NIH"].copy()
    nih_df["patient_id"] = nih_df["original_filename"].apply(lambda f: extract_patient_id(f, "NIH"))
    nih_df["abs_path"] = nih_df["final_filename"].apply(lambda p: os.path.join(DATASET_DIR, p.replace("/", os.sep)))
    nih_train, nih_val, nih_test = create_patient_splits(nih_df, SEED)
    
    cx6_df = prov_df[prov_df["original_dataset"] == "ChestX6"].copy()
    cx6_df["patient_id"] = cx6_df["original_filename"].apply(lambda f: extract_patient_id(f, "ChestX6"))
    cx6_df["abs_path"] = cx6_df["final_filename"].apply(lambda p: os.path.join(DATASET_DIR, p.replace("/", os.sep)))
    cx6_test = create_stratified_external_test(cx6_df, SEED)

    tf_tr, tf_te = get_transforms()
    dl_train = DataLoader(XRayDataset(nih_train, tf_tr), BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    dl_val   = DataLoader(XRayDataset(nih_val, tf_te), BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    dl_test  = DataLoader(XRayDataset(nih_test, tf_te), BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    dl_ext   = DataLoader(XRayDataset(cx6_test, tf_te), BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    architectures = ["resnet50", "densenet121", "efficientnet_b0"]
    all_roc_data = {}
    
    for arch in architectures:
        print(f"\n[{arch.upper()}] Starting Training...")
        model = EmphysemaClassifier(arch).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        best_val_auc = -1
        best_state = None
        
        train_losses = []
        val_aucs = []
        
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            running_loss = 0.0
            for imgs, lbls in dl_train:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, lbls)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * imgs.size(0)
                
            epoch_loss = running_loss / len(dl_train.dataset)
            train_losses.append(epoch_loss)
            
            model.eval()
            va_yt, va_ypr = [], []
            with torch.no_grad():
                for imgs, lbls in dl_val:
                    out = model(imgs.to(DEVICE))
                    probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                    va_ypr.extend(probs)
                    va_yt.extend(lbls.numpy())
                    
            ep_auc = roc_auc_score(va_yt, va_ypr)
            val_aucs.append(ep_auc)
            print(f"  Ep {epoch:02d} | Train Loss: {epoch_loss:.4f} | Val AUC: {ep_auc:.4f}")
            sys.stdout.flush()
            
            if ep_auc > best_val_auc:
                best_val_auc = ep_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Save model
        torch.save(best_state, os.path.join(OUT_DIR, f"{arch}_baseline.pth"))
        
        # Plot Training Curves
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(range(1, MAX_EPOCHS+1), train_losses, 'b-', label='Train Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Cross Entropy Loss', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax2 = ax1.twinx()
        ax2.plot(range(1, MAX_EPOCHS+1), val_aucs, 'r-', label='Validation AUC')
        ax2.set_ylabel('ROC-AUC', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        plt.title(f'Training Progress ({arch})')
        fig.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{arch}_training_curves.png"), dpi=150)
        plt.close()

        # Evaluate
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        model.eval()
        
        yt_int, ypr_int, cm_int, met_int = evaluate(model, dl_test, "INTERNAL NIH TEST SET")
        yt_ext, ypr_ext, cm_ext, met_ext = evaluate(model, dl_ext, "EXTERNAL CHESTX6 TEST SET")
        
        # Store for combined ROC plot
        all_roc_data[arch] = {
            "int_fpr": roc_curve(yt_int, ypr_int)[0], "int_tpr": roc_curve(yt_int, ypr_int)[1], "int_auc": met_int["auc"],
            "ext_fpr": roc_curve(yt_ext, ypr_ext)[0], "ext_tpr": roc_curve(yt_ext, ypr_ext)[1], "ext_auc": met_ext["auc"]
        }
        
        # Plot Confusion Matrices
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.heatmap(cm_int, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=['Normal', 'Emphysema'], yticklabels=['Normal', 'Emphysema'])
        axes[0].set_title(f'Confusion Matrix: NIH Test ({arch})')
        axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')
        
        sns.heatmap(cm_ext, annot=True, fmt='d', cmap='Greens', ax=axes[1], xticklabels=['Normal', 'Emphysema'], yticklabels=['Normal', 'Emphysema'])
        axes[1].set_title(f'Confusion Matrix: ChestX6 Test ({arch})')
        axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{arch}_confusion_matrices.png"), dpi=150)
        plt.close()

        print(f"  Final NIH Test AUC: {met_int['auc']:.4f}")
        print(f"  Final CX6 Test AUC: {met_ext['auc']:.4f}")
        sys.stdout.flush()

    # Generate Combined ROC Plot
    colors = {"resnet50": "darkorange", "densenet121": "blue", "efficientnet_b0": "purple"}
    plt.figure(figsize=(10, 8))
    for arch in architectures:
        plt.plot(all_roc_data[arch]["int_fpr"], all_roc_data[arch]["int_tpr"], color=colors[arch], lw=2, 
                 label=f'{arch} (AUC = {all_roc_data[arch]["int_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Combined ROC Curves: Internal NIH Test Set')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "combined_roc_curves.png"), dpi=150)
    plt.close()

    print("\n[DONE] All models trained and outputs saved to results/ folder.")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
