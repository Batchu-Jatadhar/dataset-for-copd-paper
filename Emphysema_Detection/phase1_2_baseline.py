import sys, os, random, time
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch, torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.model_selection import GroupShuffleSplit

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
MAX_EPOCHS  = 20
PATIENCE    = 5
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_M  = [0.485, 0.456, 0.406]; IMAGENET_S  = [0.229, 0.224, 0.225]

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

class EmphysemaDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.lbl_map = {"Normal": 0, "Emphysema": 1}
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["abs_path"]).convert("RGB")
        return self.transform(img), self.lbl_map[row["class_label"]], row["abs_path"]

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

# --- Model Factories with Grad-CAM support ---
class BaseModel(nn.Module):
    def __init__(self, model_name="resnet50"):
        super().__init__()
        self.model_name = model_name
        self.gradients = None
        if model_name == "resnet50":
            base = models.resnet50(weights="IMAGENET1K_V1")
            base.fc = nn.Linear(base.fc.in_features, 2)
            self.features = nn.Sequential(*list(base.children())[:-2])
            self.pool = base.avgpool
            self.classifier = base.fc
        elif model_name == "densenet121":
            base = models.densenet121(weights="IMAGENET1K_V1")
            base.classifier = nn.Linear(base.classifier.in_features, 2)
            self.features = base.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = base.classifier
        elif model_name == "efficientnet_b0":
            base = models.efficientnet_b0(weights="IMAGENET1K_V1")
            base.classifier[1] = nn.Linear(base.classifier[1].in_features, 2)
            self.features = base.features
            self.pool = base.avgpool
            self.classifier = base.classifier

    def activations_hook(self, grad): self.gradients = grad
    def forward(self, x):
        x = self.features(x)
        if x.requires_grad: x.register_hook(self.activations_hook)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
    def get_grad(self): return self.gradients
    def get_act(self, x): return self.features(x)

def get_gradcam(model, img_t, target_class):
    model.eval(); out = model(img_t.unsqueeze(0))
    model.zero_grad(); out[0, target_class].backward()
    grad = model.get_grad(); act = model.get_act(img_t.unsqueeze(0)).detach()
    if grad is None: return np.zeros((224,224))
    pooled_grad = torch.mean(grad, dim=[0, 2, 3])
    for i in range(act.shape[1]): act[:, i, :, :] *= pooled_grad[i]
    hm = torch.mean(act, dim=1).squeeze().cpu().numpy()
    hm = np.maximum(hm, 0)
    if np.max(hm) > 0: hm /= np.max(hm)
    return hm

def calc_metrics(yt, yp, ypr):
    acc = accuracy_score(yt, yp)
    prec = precision_score(yt, yp, zero_division=0)
    rec = recall_score(yt, yp, zero_division=0)
    f1 = f1_score(yt, yp, zero_division=0)
    auc_val = roc_auc_score(yt, ypr)
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0,1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return acc, prec, rec, f1, auc_val, sens, spec

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    
    print("==================================================")
    print(" PHASE 1: DATASET INTEGRITY & SPLITTING")
    print("==================================================")
    prov_df = pd.read_csv(PROV_CSV)
    nih_df = prov_df[prov_df["original_dataset"] == "NIH"].copy()
    nih_df["patient_id"] = nih_df["original_filename"].apply(lambda f: f.split("_")[0] if "_" in f else f)
    nih_df["abs_path"] = nih_df["final_filename"].apply(lambda p: os.path.join(DATASET_DIR, p.replace("/", os.sep)))
    
    # 70/15/15 split using GroupShuffleSplit to strictly avoid patient overlap
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    tr_idx, temp_idx = next(gss1.split(nih_df, groups=nih_df['patient_id']))
    tr_df, temp_df = nih_df.iloc[tr_idx], nih_df.iloc[temp_idx]
    
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    va_idx, te_idx = next(gss2.split(temp_df, groups=temp_df['patient_id']))
    va_df, te_df = temp_df.iloc[va_idx], temp_df.iloc[te_idx]
    
    # Assertions
    tr_p, va_p, te_p = set(tr_df["patient_id"]), set(va_df["patient_id"]), set(te_df["patient_id"])
    assert len(tr_p.intersection(va_p)) == 0, "Leakage: Train ∩ Val"
    assert len(tr_p.intersection(te_p)) == 0, "Leakage: Train ∩ Test"
    assert len(va_p.intersection(te_p)) == 0, "Leakage: Val ∩ Test"
    
    print(f"Splits -> Train: {len(tr_df)} | Val: {len(va_df)} | Test: {len(te_df)}")
    print("Assertion Passed: Train ∩ Validation patient overlap = 0")
    print("Assertion Passed: Train ∩ Test patient overlap = 0")
    print("Assertion Passed: Validation ∩ Test patient overlap = 0")
    print(f"Class Balance: Train({dict(tr_df['class_label'].value_counts())}), Val({dict(va_df['class_label'].value_counts())}), Test({dict(te_df['class_label'].value_counts())})")

    tr_ds, va_ds, te_ds = EmphysemaDataset(tr_df, tf_tr), EmphysemaDataset(va_df, tf_te), EmphysemaDataset(te_df, tf_te)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True)
    va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=True)
    te_ld = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=True)
    
    print("\n==================================================")
    print(" PHASE 1: MODEL BENCHMARKING (ResNet50, DenseNet121, EfficientNet-B0)")
    print("==================================================")
    
    models_to_test = ["resnet50", "densenet121", "efficientnet_b0"]
    bench_results = []
    histories = {}
    test_preds = {}
    
    for m_name in models_to_test:
        print(f"\n--- Training {m_name.upper()} ---")
        set_seed(42)
        model = BaseModel(m_name).to(DEVICE)
        num_params = sum(p.numel() for p in model.parameters())
        crit = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        best_val_auc = 0; patience_ctr = 0; best_ckpt = os.path.join(OUT_DIR, f"{m_name}_best.pth")
        histories[m_name] = {"tr_loss":[], "va_loss":[], "va_auc":[]}
        
        start_t = time.time()
        for ep in range(1, MAX_EPOCHS + 1):
            model.train(); run_loss = 0
            for imgs, lbls, _ in tr_ld:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                optim.zero_grad(); out = model(imgs); loss = crit(out, lbls)
                loss.backward(); optim.step()
                run_loss += loss.item() * imgs.size(0)
            tr_loss = run_loss / len(tr_ds)
            
            model.eval(); va_loss = 0; va_yt, va_ypr = [], []
            with torch.no_grad():
                for imgs, lbls, _ in va_ld:
                    out = model(imgs.to(DEVICE)); loss = crit(out, lbls.to(DEVICE))
                    va_loss += loss.item() * imgs.size(0)
                    va_ypr.extend(torch.softmax(out, 1)[:,1].cpu().numpy()); va_yt.extend(lbls.numpy())
            va_loss = va_loss / len(va_ds)
            va_auc = roc_auc_score(va_yt, va_ypr)
            histories[m_name]["tr_loss"].append(tr_loss); histories[m_name]["va_loss"].append(va_loss); histories[m_name]["va_auc"].append(va_auc)
            
            print(f"  Ep {ep} | Tr Loss: {tr_loss:.4f} | Va Loss: {va_loss:.4f} | Va AUC: {va_auc:.4f}")
            if va_auc > best_val_auc:
                best_val_auc = va_auc; patience_ctr = 0
                torch.save(model.state_dict(), best_ckpt)
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    print(f"  Early stopping at epoch {ep}!")
                    break
        train_time = time.time() - start_t
        
        # Eval on Test Set
        model.load_state_dict(torch.load(best_ckpt))
        model.eval(); yt, yp, ypr, paths = [], [], [], []
        with torch.no_grad():
            for imgs, lbls, p in te_ld:
                out = model(imgs.to(DEVICE)); pr = torch.softmax(out, 1)
                yp.extend(pr.argmax(1).cpu().numpy()); yt.extend(lbls.numpy()); ypr.extend(pr[:,1].cpu().numpy())
                paths.extend(p)
        yt, yp, ypr = np.array(yt), np.array(yp), np.array(ypr)
        test_preds[m_name] = (yt, yp, ypr, paths)
        
        acc, prec, rec, f1, auc_val, sens, spec = calc_metrics(yt, yp, ypr)
        bench_results.append({
            "Model": m_name, "Parameters": num_params, "Train_Time_s": train_time,
            "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "ROC-AUC": auc_val, "Sens": sens, "Spec": spec
        })
        print(f"  Test -> Acc: {acc:.4f} | AUC: {auc_val:.4f} | F1: {f1:.4f}")

    df_bench = pd.DataFrame(bench_results)
    df_bench.to_csv(os.path.join(OUT_DIR, "benchmark_models.csv"), index=False)
    best_model_name = df_bench.sort_values(by="ROC-AUC", ascending=False).iloc[0]["Model"]
    print(f"\nBest Backbone Identified: {best_model_name.upper()}")

    # PLOTS
    plt.figure(figsize=(15, 5))
    for i, m_name in enumerate(models_to_test):
        plt.subplot(1, 3, i+1)
        yt, yp, _, _ = test_preds[m_name]
        sns.heatmap(confusion_matrix(yt, yp), annot=True, fmt='d', cmap='Blues')
        plt.title(f"{m_name} CM"); plt.ylabel('True'); plt.xlabel('Pred')
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "all_cm.png")); plt.close()
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for m_name in models_to_test:
        yt, _, ypr, _ = test_preds[m_name]
        fpr, tpr, _ = roc_curve(yt, ypr)
        plt.plot(fpr, tpr, label=f"{m_name} (AUC={roc_auc_score(yt,ypr):.3f})")
    plt.plot([0,1], [0,1], 'k--'); plt.title("ROC Curves"); plt.legend()
    plt.subplot(1, 2, 2)
    for m_name in models_to_test:
        yt, _, ypr, _ = test_preds[m_name]
        precision, recall, _ = precision_recall_curve(yt, ypr)
        plt.plot(recall, precision, label=m_name)
    plt.title("Precision-Recall Curves"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "roc_pr_curves.png")); plt.close()

    plt.figure(figsize=(15, 5))
    for i, m_name in enumerate(models_to_test):
        plt.subplot(1, 3, i+1)
        plt.plot(histories[m_name]["tr_loss"], label='Train Loss')
        plt.plot(histories[m_name]["va_loss"], label='Val Loss')
        plt.title(f"{m_name} Learning Curve"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "training_curves.png")); plt.close()

    print("\n==================================================")
    print(f" PHASE 1: EXPLAINABILITY AUDIT (Grad-CAM for {best_model_name})")
    print("==================================================")
    best_model = BaseModel(best_model_name).to(DEVICE)
    best_model.load_state_dict(torch.load(os.path.join(OUT_DIR, f"{best_model_name}_best.pth")))
    yt, yp, ypr, paths = test_preds[best_model_name]
    
    tp_idx = next((i for i, (t, p) in enumerate(zip(yt, yp)) if t==1 and p==1), -1)
    tn_idx = next((i for i, (t, p) in enumerate(zip(yt, yp)) if t==0 and p==0), -1)
    fp_idx = next((i for i, (t, p) in enumerate(zip(yt, yp)) if t==0 and p==1), -1)
    fn_idx = next((i for i, (t, p) in enumerate(zip(yt, yp)) if t==1 and p==0), -1)
    cases = {"True Positive": tp_idx, "True Negative": tn_idx, "False Positive": fp_idx, "False Negative": fn_idx}
    
    inv_norm = transforms.Normalize([-m/s for m,s in zip(IMAGENET_M,IMAGENET_S)], [1/s for s in IMAGENET_S])
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (name, idx) in zip(axes, cases.items()):
        if idx == -1: ax.set_title(f"{name} (None Found)"); ax.axis("off"); continue
        path = paths[idx]; img = Image.open(path).convert("RGB")
        img_t = tf_te(img).to(DEVICE); img_t.requires_grad_()
        hm = get_gradcam(best_model, img_t, target_class=1)
        hm = cv2.resize(hm, (224, 224)); hm_colored = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
        hm_colored = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)
        orig_img = np.uint8(255 * np.clip(inv_norm(img_t.detach().cpu()).permute(1, 2, 0).numpy(), 0, 1))
        ax.imshow(cv2.addWeighted(orig_img, 0.5, hm_colored, 0.5, 0))
        ax.set_title(name); ax.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, f"{best_model_name}_gradcam.png")); plt.close()

    print("\n==================================================")
    print(f" PHASE 1: REPRODUCIBILITY (3 Seeds for {best_model_name})")
    print("==================================================")
    rep_metrics = []
    seeds = [100, 2024, 42] # we already ran 42, but we'll re-run just for clean loop or use existing
    for s in seeds:
        print(f"  Seed {s}...")
        set_seed(s)
        model = BaseModel(best_model_name).to(DEVICE)
        optim = torch.optim.Adam(model.parameters(), lr=1e-4)
        crit = nn.CrossEntropyLoss()
        best_va_auc = 0; best_w = None; pat = 0
        for ep in range(1, MAX_EPOCHS + 1):
            model.train()
            for imgs, lbls, _ in tr_ld:
                optim.zero_grad(); out = model(imgs.to(DEVICE))
                loss = crit(out, lbls.to(DEVICE)); loss.backward(); optim.step()
            model.eval(); va_yt, va_ypr = [], []
            with torch.no_grad():
                for imgs, lbls, _ in va_ld:
                    out = model(imgs.to(DEVICE))
                    va_ypr.extend(torch.softmax(out, 1)[:,1].cpu().numpy()); va_yt.extend(lbls.numpy())
            va_auc = roc_auc_score(va_yt, va_ypr)
            if va_auc > best_va_auc: best_va_auc = va_auc; best_w = model.state_dict(); pat = 0
            else:
                pat += 1
                if pat >= PATIENCE: break
        
        model.load_state_dict(best_w); model.eval()
        yt, yp, ypr = [], [], []
        with torch.no_grad():
            for imgs, lbls, _ in te_ld:
                out = model(imgs.to(DEVICE)); pr = torch.softmax(out, 1)
                yp.extend(pr.argmax(1).cpu().numpy()); yt.extend(lbls.numpy()); ypr.extend(pr[:,1].cpu().numpy())
        acc, prec, rec, f1, auc_val, _, _ = calc_metrics(yt, yp, ypr)
        rep_metrics.append([acc, prec, rec, f1, auc_val])
    
    met = np.array(rep_metrics)
    print(f"  Reproducibility Mean ± Std:")
    print(f"  Acc: {met[:,0].mean():.4f} ± {met[:,0].std():.4f} | AUC: {met[:,4].mean():.4f} ± {met[:,4].std():.4f}")

    print("\n==================================================")
    print(" PHASE 2: EXTERNAL VALIDATION (ChestX6)")
    print("==================================================")
    cx6_df = prov_df[prov_df["original_dataset"] == "ChestX6"].copy()
    cx6_df["abs_path"] = cx6_df["final_filename"].apply(lambda p: os.path.join(DATASET_DIR, p.replace("/", os.sep)))
    ex_ds = EmphysemaDataset(cx6_df, tf_te)
    ex_ld = DataLoader(ex_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=True)
    
    best_model.eval(); e_yt, e_yp, e_ypr = [], [], []
    with torch.no_grad():
        for imgs, lbls, _ in ex_ld:
            out = best_model(imgs.to(DEVICE)); pr = torch.softmax(out, 1)
            e_yp.extend(pr.argmax(1).cpu().numpy()); e_yt.extend(lbls.numpy()); e_ypr.extend(pr[:,1].cpu().numpy())
    e_acc, e_prec, e_rec, e_f1, e_auc, e_sens, e_spec = calc_metrics(e_yt, e_yp, e_ypr)
    
    ext_results = pd.DataFrame([{
        "Dataset": "ChestX6", "Accuracy": e_acc, "Precision": e_prec, "Recall": e_rec, 
        "F1": e_f1, "ROC-AUC": e_auc, "Sens": e_sens, "Spec": e_spec
    }])
    ext_results.to_csv(os.path.join(OUT_DIR, "external_validation.csv"), index=False)
    
    print("ChestX6 Performance:")
    print(f"  Acc: {e_acc:.4f} | Prec: {e_prec:.4f} | Rec: {e_rec:.4f} | F1: {e_f1:.4f} | AUC: {e_auc:.4f}")
    print("\nPhase 1 & 2 Completed Successfully.")
