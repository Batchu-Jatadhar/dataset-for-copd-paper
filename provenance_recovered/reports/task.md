# DANN Validation Tasks

- `[x]` **Phase 1: Shortcut Mitigation Verification**
  - `[x]` Implement robust OpenCV lung masking.
  - `[x]` Train Strong DANN on Lung-Masked data.
  - `[x]` Train Random-Label DANN on Lung-Masked data.
  - `[x]` Verify random-label AUC drops to ~0.50. (Failed: AUC=0.7074)

- `[x]` **Phase 1.5: Preprocessing Ablation Study (Acquisition Artifact Mitigation)**
  - `[x]` Implement 4 preprocessing transforms: Histogram Matching, Z-score, CLAHE+Hist, Frequency Suppression.
  - `[x]` Run Real-Label DANN for each to measure Disease AUC and Domain Acc.
  - `[x]` Run Random-Label DANN for each to measure Random-label AUC.
  - `[x]` Identify the transform that collapses Domain Acc (~50%) and Random AUC (~0.50). (Failed: Lowest AUC=0.8193 with CLAHE).

- `[-]` **Phase 2: Final DANN & Feature Attribution** *(Canceled)*
  - `[-]` Ensure Domain Accuracy approaches 45-60% (Strong GRL).
  - `[-]` Implement Grad-CAM.
  - `[-]` Generate and review heatmaps for valid activation fields.
