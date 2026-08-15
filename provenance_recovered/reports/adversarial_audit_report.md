# Adversarial Validation Audit Report
## DANN Emphysema Detection — ROC-AUC 0.9999 Defensibility Analysis

> **Audit Completed:** 2026-06-13 | **Device:** CUDA GPU | **Runtime:** ~30 min
> **Script:** [`adversarial_audit.py`](file:///c:/Users/batch_dcahfiw/Downloads/research%20paper%202/Emphysema_Detection/adversarial_audit.py)
> **Output Dir:** `results_final_study/adversarial_audit/`

---

## Executive Summary

| Check | Description | Result | Verdict |
|-------|-------------|--------|---------|
| **1** | Multi-Seed Reproducibility (3 seeds) | AUC = 0.9999 ± 0.0000 | ✅ PASS |
| **2** | ChestX6 Hold-Out Leakage | Patient overlap = 0, File overlap = 0 | ✅ PASS |
| **3** | Domain Classifier Audit | NIH domain acc=1.00, CX6 domain acc=0.37 | ⚠️ WARN |
| **4** | Random Label Sanity Check | Random-trained model AUC = **0.9970** | 🚨 CRITICAL FAIL |
| **5** | Grad-CAM Attribution | Weights not found — visual invalid | ⚠️ MANUAL |
| **6** | Calibration Analysis | 22.5% near-0, 11.9% near-1 | ✅ PASS |
| **7** | **Final Verdict** | — | ❌ **NOT FULLY DEFENSIBLE** |

---

## Check 1 — Multi-Seed Reproducibility

**Seeds tested:** 42, 123, 2024 | **Epochs:** 10 | **Backbone:** ResNet-50 DANN

| Seed | Accuracy | Precision | Recall | F1 | ROC-AUC |
|------|----------|-----------|--------|----|---------|
| 42 | 86.52% | **100.00%** | 83.06% | 90.75% | 0.99992 |
| 123 | 92.29% | **100.00%** | 90.31% | 94.91% | 0.99990 |
| 2024 | 88.95% | **100.00%** | 86.12% | 92.54% | 0.99996 |
| **Mean** | **89.25%** | **100.00%** | **86.50%** | **92.73%** | **0.99993** |
| **Std** | ±2.90% | ±0.00% | ±3.64% | ±2.09% | **±0.00003** |

> **Finding:** AUC is essentially constant at 0.9999 with near-zero variance across all three seeds. This rules out a lucky random initialization as an explanation for the high performance. The signal is real and reproducible.

> [!TIP]
> The 100% Precision across all seeds means the model **never produces a false positive** on CX6. This is itself suspicious and worth investigating — it often indicates the model is exploiting a feature that cleanly separates domains rather than true disease morphology.

---

## Check 2 — ChestX6 Hold-Out Leakage Verification

```
Train patients  (NIH train + CX6 train): 5,144 unique patient IDs
Test patients   (CX6 test only)        :   655 unique patient IDs
Train ∩ Test overlap (patient-level)   :     0   ✅
Train ∩ Test overlap (file-level)      :     0   ✅
```

> **Finding:** No leakage at the patient or image level. The GroupShuffleSplit with `random_state=42` correctly partitions patients. The high AUC is **not** caused by test images appearing in training.

---

## Check 3 — Domain Classifier Audit

```
NIH  Test Set:  Domain Accuracy = 1.0000  |  %pred-NIH = 100.0%  |  %pred-CX6 = 0.0%
CX6  Test Set:  Domain Accuracy = 0.3727  |  %pred-NIH =  62.7%  |  %pred-CX6 = 37.3%
```

> [!WARNING]
> **NIH domain accuracy = 1.0000** — The domain classifier perfectly identifies every NIH image as NIH. This means the GRL **completely failed to confuse the domain classifier** for the source (NIH) domain. The backbone is still producing domain-discriminative features.

> **CX6 domain accuracy = 0.37** (below chance) — The model predicts 62.7% of CX6 images as NIH. This is partial confusion but it's asymmetric — the DANN did not achieve balanced domain invariance.

**Why this matters:** A properly functioning DANN should produce domain accuracy ≈ 50% for *both* domains. The collapse on NIH (100%) strongly suggests the alpha scaling (`max α = 0.1`) was too weak to overcome the source-domain signal. Domain-invariant features were only partially learned.

📊 See: [`check3_domain_cm.png`](file:///c:/Users/batch_dcahfiw/Downloads/research%20paper%202/results_final_study/adversarial_audit/check3_domain_cm.png)

---

## Check 4 — Random Label Sanity Check 🚨

Two sub-experiments were run:

### 4a — Real model evaluated on shuffled test labels
```
Shuffled-label Accuracy : 62.17%
Shuffled-label ROC-AUC  : 0.5064  ✅  (expected ≈ 0.50)
```
The trained model gives random predictions when ground truth is shuffled → the model output is correlated with real labels, not arbitrary structure.

### 4b — Model TRAINED FROM SCRATCH on shuffled training labels, then tested on real labels
```
Random-Trained Accuracy : 44.32%
Random-Trained ROC-AUC  : 0.9970  🚨  (expected ≈ 0.50)
```

> [!CAUTION]
> **This is the critical finding of the entire audit.** A model trained with completely random disease labels still achieves **AUC = 0.9970** on the real test set. This is only possible if there exists a **dataset-level shortcut feature** that:
> - Is correlated with the ChestX6 vs NIH split
> - Is visible to the model even under random label training
> - Produces predictions that happen to align with real emphysema labels in the test set

**Most likely explanation:** The ChestX6 dataset's emphysema images come from a systematically different patient population or acquisition protocol than Normal images. The model is detecting an **imaging style / dataset signature**, not emphysema pathology. Since the disease prevalence is also skewed differently per domain, training on random NIH labels still produces features that accidentally correlate with CX6 emphysema labels.

---

## Check 5 — Feature Attribution Verification (Grad-CAM)

> [!WARNING]
> The Phase-1 ResNet-50 best checkpoint (`resnet50_best.pth`) was not found in `results_final_study/`. Grad-CAM was run with random weights and the output is therefore **not valid for clinical interpretation**.

📊 See: [`check5_gradcam.png`](file:///c:/Users/batch_dcahfiw/Downloads/research%20paper%202/results_final_study/adversarial_audit/check5_gradcam.png) — ⚠️ **Invalid (random weights)**

**Action required:** Re-run with the correct checkpoint path. The trained model file is at:
- `results_final_study/dann_best.pth` (DANN model)
- Or re-run Phase 1 to regenerate `resnet50_best.pth`

---

## Check 6 — Calibration Analysis

```
Predictions < 0.05 :  22.5%
Predictions > 0.95 :  11.9%
Total saturated    :  34.4%   (threshold: >90% = fail)
```

> **Finding:** 34.4% of predictions are in the high-confidence zone, but 65.6% of predictions are in the intermediate range (0.05–0.95). This is **not** a collapsed/degenerate probability distribution. The model is producing a spread of confidence levels, suggesting it has genuine uncertainty on many cases.

📊 See: [`check6_calibration.png`](file:///c:/Users/batch_dcahfiw/Downloads/research%20paper%202/results_final_study/adversarial_audit/check6_calibration.png)

---

## Check 7 — Final Verdict

### Can the 0.9999 ROC-AUC be defended during peer review?

## ❌ NOT FULLY DEFENSIBLE in its current form

The following critical issue must be resolved before the result can be published:

### 🚨 Primary Concern: Shortcut Feature Contamination (Check 4b)

A model trained on **random labels** achieving **AUC=0.9970** is definitive evidence that the network is exploiting a dataset-level artefact, not emphysema pathology. The most probable causes are:

| Possible Shortcut | Likelihood | Evidence |
|---|---|---|
| **Dataset-of-origin signature** (scanner/acquisition style difference between CX6 Normal vs Emphysema) | HIGH | Domain acc collapse on NIH (Check 3), DANN failed to fully equalize distributions |
| **Prevalence imbalance across domains** (CX6 emphysema images sampled from a different age/severity stratum) | MEDIUM | 100% Precision on CX6 across all seeds |
| **Filename/metadata leakage** (label embedded in path prefix used during loading) | LOW | Check 2 passed at patient/file level, but label-in-filename not tested |
| **Image artifact differences** (lung hyperinflation in emphysema → systematically larger lung fields detectable by border statistics) | MEDIUM | Grad-CAM check inconclusive (invalid weights) |

### ⚠️ Secondary Concern: Incomplete Domain Adaptation (Check 3)

The DANN's domain classifier retained 100% accuracy on NIH despite the GRL. This means features remain partially domain-discriminative — the GRL alpha cap of 0.1 was too conservative.

---

## Recommended Remediation Steps

> [!IMPORTANT]
> These steps are **required** before the result can be submitted to peer review.

### 1. Identify and Remove the Shortcut

```python
# Test: Train ONLY on CX6 data (no NIH at all). 
# If AUC remains near 1.0 with random labels, the shortcut is within CX6 itself
# (e.g., Normal vs Emphysema images differ in acquisition protocol).

# Test: Check whether lung segmentation masks differ systematically:
# emphysema patients often have hyperinflated, barrel-shaped chests.
# Use a lung segmentation model (e.g. U-Net) and measure lung area ratio.
```

### 2. Lung-Cropped Re-evaluation

Train and test exclusively on **lung-segmented ROIs** (mask out everything outside the lung boundary). If AUC drops substantially, the shortcut was in the non-lung region (borders, text, markers).

### 3. Strengthen the DANN GRL

```python
# Increase alpha ceiling from 0.1 to 1.0
alpha = 2. / (1. + np.exp(-10 * p)) - 1   # full range [0, 1]
# Monitor until NIH domain accuracy drops to 55–60%
```

### 4. Fix Grad-CAM

```python
# Point Grad-CAM to the DANN checkpoint:
gc_model.load_state_dict(torch.load(
    r"results_final_study/dann_best.pth", map_location=DEVICE
))
# Visually verify: attention must be on lung parenchyma, not image edges or corners
```

### 5. Cross-Dataset Negative Control

Mix a 3rd unseen dataset (e.g., NIH ChestX-14 Normal-only cases) into the CX6 test set. If AUC drops dramatically, the model is recognizing datasets, not emphysema.

---

## Data Summary

| Split | Dataset | Patients | Images |
|-------|---------|----------|--------|
| Train | NIH | — | 4,022 |
| Test | NIH | — | 1,010 |
| Train | ChestX6 | — | 2,616 |
| **Test** | **ChestX6** | **655** | **3,204** |

---

*Report generated by `adversarial_audit.py` — 2026-06-13*
