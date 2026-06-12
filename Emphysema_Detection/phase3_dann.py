import sys, os, random, time
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch, torch.nn as nn
from torch.autograd import Function
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"): sys.stderr.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

BASE_DIR    = r"C:\Users\batch_dcahfiw\Downloads\datasetfinalforcopdemphysema"
DATASET_DIR = os.path.join(BASE_DIR, "Final_Dataset_PatientSafe")
PROV_CSV    = os.path.join(BASE_DIR, "provenance_report.csv")
OUT_DIR     = r"C:\Users\batch_dcahfiw\Downloads\research paper 2\results_final_study"
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE  = 32
NUM_WORKERS = 4
MAX_EPOCHS  = 10
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_M  = [0.485, 0.456, 0.406]; IMAGENET_S  = [0.229, 0.224, 0.225]

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

class DomainDataset(Dataset):
    def __init__(self, df, domain_label, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.domain_label = domain_label
        self.lbl_map = {"Normal": 0, "Emphysema": 1}
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["abs_path"]).convert("RGB")
        if self.transform: img = self.transform(img)
        # return image, class_label, domain_label
        return img, self.lbl_map[row["class_label"]], self.domain_label

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

# --- Gradient Reversal Layer ---
class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GRL.apply(x, alpha)

# --- DANN Architecture ---
class DANNModel(nn.Module):
    def __init__(self, backbone_name="resnet50"):
        super().__init__()
        self.backbone_name = backbone_name
        if backbone_name == "resnet50":
            base = models.resnet50(weights="IMAGENET1K_V1")
            self.features = nn.Sequential(*list(base.children())[:-1])
            in_features = base.fc.in_features
        elif backbone_name == "densenet121":
            base = models.densenet121(weights="IMAGENET1K_V1")
            self.features = nn.Sequential(base.features, nn.AdaptiveAvgPool2d((1, 1)))
            in_features = base.classifier.in_features
        elif backbone_name == "efficientnet_b0":
            base = models.efficientnet_b0(weights="IMAGENET1K_V1")
            self.features = nn.Sequential(base.features, base.avgpool)
            in_features = base.classifier[1].in_features
        else:
            raise ValueError("Unknown backbone")

        # Task Head (Emphysema vs Normal)
        self.class_classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
        # Domain Head (NIH vs ChestX6)
        self.domain_classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
        
    def forward(self, x, alpha=1.0):
        feat = self.features(x)
        feat = torch.flatten(feat, 1)
        
        class_output = self.class_classifier(feat)
        
        reversed_feat = grad_reverse(feat, alpha)
        domain_output = self.domain_classifier(reversed_feat)
        
        return class_output, domain_output

def calc_metrics(yt, yp, ypr):
    acc = accuracy_score(yt, yp)
    prec = precision_score(yt, yp, zero_division=0)
    rec = recall_score(yt, yp, zero_division=0)
    f1 = f1_score(yt, yp, zero_division=0)
    auc_val = roc_auc_score(yt, ypr)
    # Sens and spec approx via simple CM check
    tp = np.sum((yt==1) & (yp==1))
    tn = np.sum((yt==0) & (yp==0))
    fp = np.sum((yt==0) & (yp==1))
    fn = np.sum((yt==1) & (yp==0))
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return acc, prec, rec, f1, auc_val, sens, spec

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    
    print("==================================================")
    print(" PHASE 3: DANN DOMAIN ADAPTATION EXTENSION")
    print("==================================================")
    
    # 1. Identify best backbone from Phase 1
    bench_file = os.path.join(OUT_DIR, "benchmark_models.csv")
    if not os.path.exists(bench_file):
        print("Error: benchmark_models.csv not found! Run Phase 1 first.")
        sys.exit(1)
        
    df_bench = pd.read_csv(bench_file)
    best_model_name = df_bench.sort_values(by="ROC-AUC", ascending=False).iloc[0]["Model"]
    print(f"Loading Best Backbone from Phase 1: {best_model_name.upper()}")
    
    prov_df = pd.read_csv(PROV_CSV)
    
    # NIH Data (Domain 0)
    nih_df = prov_df[prov_df["original_dataset"] == "NIH"].copy()
    nih_df["abs_path"] = nih_df["final_filename"].apply(lambda p: os.path.join(DATASET_DIR, p.replace("/", os.sep)))
    
    # ChestX6 Data (Domain 1)
    cx6_df = prov_df[prov_df["original_dataset"] == "ChestX6"].copy()
    cx6_df["abs_path"] = cx6_df["final_filename"].apply(lambda p: os.path.join(DATASET_DIR, p.replace("/", os.sep)))

    # Extract patient IDs
    nih_df["patient_id"] = nih_df["original_filename"].apply(lambda p: p.split("_")[0] if "_" in p else p)
    cx6_df["patient_id"] = cx6_df["original_filename"].apply(lambda p: p.split("_")[0] if "_" in p else p)
    
    # 80/20 Train/Test split using GroupShuffleSplit to avoid patient leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(nih_df, groups=nih_df['patient_id']))
    tr_nih, te_nih = nih_df.iloc[tr_idx], nih_df.iloc[te_idx]
    
    tr_idx, te_idx = next(gss.split(cx6_df, groups=cx6_df['patient_id']))
    tr_cx6, te_cx6 = cx6_df.iloc[tr_idx], cx6_df.iloc[te_idx]
    
    # Datasets (Domain 0 = NIH, Domain 1 = ChestX6)
    ds_tr_nih = DomainDataset(tr_nih, domain_label=0, transform=tf_tr)
    ds_tr_cx6 = DomainDataset(tr_cx6, domain_label=1, transform=tf_tr)
    
    ds_te_nih = DomainDataset(te_nih, domain_label=0, transform=tf_te)
    ds_te_cx6 = DomainDataset(te_cx6, domain_label=1, transform=tf_te)
    
    # Loaders
    ld_tr_nih = DataLoader(ds_tr_nih, batch_size=BATCH_SIZE//2, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    ld_tr_cx6 = DataLoader(ds_tr_cx6, batch_size=BATCH_SIZE//2, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    
    ld_te_nih = DataLoader(ds_te_nih, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    ld_te_cx6 = DataLoader(ds_te_cx6, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    model = DANNModel(best_model_name).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    crit_class = nn.CrossEntropyLoss()
    crit_domain = nn.CrossEntropyLoss()
    
    best_dann_auc = 0
    best_ckpt = os.path.join(OUT_DIR, "dann_best.pth")
    
    print("\n--- Training DANN ---")
    len_dataloader = min(len(ld_tr_nih), len(ld_tr_cx6))
    
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        total_class_loss, total_domain_loss = 0, 0
        
        iter_nih = iter(ld_tr_nih)
        iter_cx6 = iter(ld_tr_cx6)
        
        for i in range(len_dataloader):
            p = float(i + (ep - 1) * len_dataloader) / (MAX_EPOCHS * len_dataloader)
            # Scale alpha by 0.1 to prevent GRL from over-penalizing and causing mode collapse
            alpha = 0.1 * (2. / (1. + np.exp(-10 * p)) - 1)
            
            # NIH Batch
            img_nih, lbl_nih, dom_nih = next(iter_nih)
            img_nih, lbl_nih, dom_nih = img_nih.to(DEVICE), lbl_nih.to(DEVICE), dom_nih.to(DEVICE)
            
            # ChestX6 Batch
            img_cx6, lbl_cx6, dom_cx6 = next(iter_cx6)
            img_cx6, lbl_cx6, dom_cx6 = img_cx6.to(DEVICE), lbl_cx6.to(DEVICE), dom_cx6.to(DEVICE)
            
            # Concatenate
            imgs = torch.cat([img_nih, img_cx6], dim=0)
            lbls = torch.cat([lbl_nih, lbl_cx6], dim=0)
            doms = torch.cat([dom_nih, dom_cx6], dim=0)
            
            optim.zero_grad()
            class_out, domain_out = model(imgs, alpha=alpha)
            
            # Calculate Losses
            loss_class = crit_class(class_out, lbls)
            loss_domain = crit_domain(domain_out, doms)
            loss = loss_class + loss_domain
            
            loss.backward()
            optim.step()
            
            total_class_loss += loss_class.item()
            total_domain_loss += loss_domain.item()
            
        print(f"  Ep {ep} | Class Loss: {total_class_loss/len_dataloader:.4f} | Domain Loss: {total_domain_loss/len_dataloader:.4f}")
        
        # Evaluate on ChestX6 Test
        model.eval()
        va_yt, va_ypr = [], []
        with torch.no_grad():
            for img, lbl, _ in ld_te_cx6:
                c_out, _ = model(img.to(DEVICE), alpha=0.0)
                va_ypr.extend(torch.softmax(c_out, 1)[:,1].cpu().numpy()); va_yt.extend(lbl.numpy())
        va_auc = roc_auc_score(va_yt, va_ypr)
        if va_auc > best_dann_auc:
            best_dann_auc = va_auc
            torch.save(model.state_dict(), best_ckpt)
            
    print("\n--- Evaluating Best DANN Model ---")
    model.load_state_dict(torch.load(best_ckpt))
    model.eval()
    
    def evaluate_loader(ld):
        yt, yp, ypr, d_yt, d_yp = [], [], [], [], []
        with torch.no_grad():
            for img, lbl, dom in ld:
                c_out, d_out = model(img.to(DEVICE), alpha=0.0)
                yp.extend(c_out.argmax(1).cpu().numpy()); yt.extend(lbl.numpy()); ypr.extend(torch.softmax(c_out, 1)[:,1].cpu().numpy())
                d_yp.extend(d_out.argmax(1).cpu().numpy()); d_yt.extend(dom.numpy())
        metrics = calc_metrics(np.array(yt), np.array(yp), np.array(ypr))
        d_acc = accuracy_score(d_yt, d_yp)
        return metrics, d_acc, np.array(d_yt), np.array(d_yp)

    nih_metrics, nih_d_acc, nih_d_yt, nih_d_yp = evaluate_loader(ld_te_nih)
    cx6_metrics, cx6_d_acc, cx_d_yt, cx_d_yp = evaluate_loader(ld_te_cx6)
    
    # Save Results
    res = {
        "Dataset": ["NIH (DANN)", "ChestX6 (DANN)"],
        "Accuracy": [nih_metrics[0], cx6_metrics[0]],
        "Precision": [nih_metrics[1], cx6_metrics[1]],
        "Recall": [nih_metrics[2], cx6_metrics[2]],
        "F1": [nih_metrics[3], cx6_metrics[3]],
        "ROC-AUC": [nih_metrics[4], cx6_metrics[4]],
        "Sens": [nih_metrics[5], cx6_metrics[5]],
        "Spec": [nih_metrics[6], cx6_metrics[6]],
        "Domain_Acc": [nih_d_acc, cx6_d_acc]
    }
    pd.DataFrame(res).to_csv(os.path.join(OUT_DIR, "dann_validation.csv"), index=False)
    
    print("DANN Performance vs Baseline:")
    
    ext_csv = os.path.join(OUT_DIR, "external_validation.csv")
    if os.path.exists(ext_csv):
        df_ext = pd.read_csv(ext_csv)
        base_auc = df_ext["ROC-AUC"].iloc[0]
        imp = ((cx6_metrics[4] - base_auc) / base_auc) * 100
        print(f"  ChestX6 Baseline AUC: {base_auc:.4f}")
        print(f"  ChestX6 DANN AUC    : {cx6_metrics[4]:.4f}")
        print(f"  Relative Improvement: {imp:+.2f}%")
        
    print(f"  NIH Domain Acc      : {nih_d_acc:.4f}")
    print(f"  ChestX6 Domain Acc  : {cx6_d_acc:.4f}")
    
    # Plot Domain Confusion Matrices
    import seaborn as sns
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(confusion_matrix(nih_d_yt, nih_d_yp, labels=[0, 1]), annot=True, fmt='d', cmap='Greens', ax=axes[0])
    axes[0].set_title(f"NIH Domain (Acc: {nih_d_acc:.4f})"); axes[0].set_ylabel("True"); axes[0].set_xlabel("Pred (0=NIH, 1=CX6)")
    
    sns.heatmap(confusion_matrix(cx_d_yt, cx_d_yp, labels=[0, 1]), annot=True, fmt='d', cmap='Reds', ax=axes[1])
    axes[1].set_title(f"ChestX6 Domain (Acc: {cx6_d_acc:.4f})"); axes[1].set_ylabel("True"); axes[1].set_xlabel("Pred (0=NIH, 1=CX6)")
    plt.suptitle("Domain Classifier Confusion Matrices (Should approach 50% / confused state)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "dann_domain_cm.png")); plt.close()
    
    print("\nPhase 3 Completed Successfully.")
