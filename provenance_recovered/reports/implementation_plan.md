# DANN Validation and Shortcut Mitigation Plan

The goal is to force the model to learn genuine emphysema pathology instead of dataset-specific artifacts, and strictly validate its behavior using 5 requested experiments.

## Proposed Changes

We will create a new self-contained script `phase_dann_validation.py` inside `Emphysema_Detection` to execute all 5 experiments in one go.

### 1. Strong DANN with Alpha=1.0
- **Change:** Update the Gradient Reversal Layer (GRL) schedule to the standard formula: `alpha = 2.0 / (1.0 + exp(-10 * p)) - 1.0`.
- **Purpose:** Provide a sufficiently strong gradient to the backbone to genuinely confuse the domain classifier, driving domain accuracy down from 0.9983 to ~50%.
- **Monitoring:** Track Disease AUC, Domain AUC, and Domain Accuracy continuously per epoch.

### 2. Lung-Only Training via Morphological Masking
- **Context:** We do not have a pre-trained UNet segmentation model available in the workspace.
- **Change:** Implement an unsupervised computer-vision lung masking pipeline using OpenCV.
  - Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
  - Use Otsu/K-means thresholding + morphological operations (erosion/dilation) to isolate the central radiolucent (dark) lung regions and remove borders, text annotations, and background.
  - The model will be trained and evaluated strictly on these masked images.
- **Purpose:** Eliminate acquisition-source artifacts (like border artifacts, background brightness) that the random-label experiment proved were being exploited.

### 3. Random Label Verification
- **Change:** Re-train the Strong DANN on the Lung-Masked images using **shuffled disease labels**.
- **Target:** AUC must collapse toward ~0.50. This is the definitive proof that the dataset-level shortcut has been successfully mitigated by the lung masking.

### 4. Domain Classifier Monitoring
- **Change:** Extend the training loop logs and final results table to report Disease AUC, Domain AUC, and Domain Accuracy side-by-side.

### 5. Feature Attribution (Grad-CAM)
- **Change:** Implement a custom PyTorch Grad-CAM extractor.
  - Attach hooks to the final convolutional layer of the ResNet-50 backbone (`layer4`).
  - Generate class-activation heatmaps for both the genuine label (Emphysema) and the domain label.
  - Output visualizations overlaid on the original and masked X-rays.
- **Purpose:** Visually confirm that the model's decision-making is localized within the lung fields and not looking at image borders or corners.

## User Review Required

> [!WARNING]
> **Lung Segmentation Approach**
> Since we do not have a pre-trained lung segmentation model (e.g., UNet) locally or via a library like `monai`, the plan proposes using an **OpenCV morphological approach** (thresholding + contour filtering). While this is usually effective at removing obvious border artifacts and outside-body background, it is less precise than deep-learning segmentation. 
> 
> **Question:** Is an unsupervised OpenCV-based lung mask acceptable for this experiment, or do you have a path to a pre-trained lung segmentation model that I should use instead?

## Verification Plan

1. **Domain Accuracy:** Must drop from ~0.99 to 45-60%.
2. **Random Label AUC:** Must drop from ~0.87 to ~0.50.
3. **Disease AUC:** Should remain defensible (>0.90 on NIH, and theoretically similar on ChestX6).
4. **Grad-CAM:** Heatmaps saved to `results_final_study/validation/` must show activation focused within the lung region.
