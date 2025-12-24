# COVID-19 Detection from Chest X-ray using EfficientNetB7 and Explainable AI

This project implements an end-to-end deep learning pipeline for binary classification of COVID-19 vs Normal chest X-ray images using EfficientNetB7. The model is enhanced with explainable AI techniques, including Grad-CAM++ for visual explanations and NAOPC (Normalized Area Over Perturbation Curve) for quantitative evaluation of explanation quality.

---

## Features

- EfficientNetB7 backbone with mixed precision (float16)
- Binary classification: COVID vs Normal
- Stratified train-test split
- Grad-CAM++ visual explanations
- NAOPC quantitative explainability evaluation
- Confusion matrix and full metric reporting
- Automatic saving of results and plots

---

## Dataset

- Dataset: COVID-19 Radiography Dataset
- Classes:
  - COVID
  - Normal
- Image preprocessing:
  - Resized to 224×224
  - Normalized to [0, 1]
- Maximum of 10,700 images per class used

---

## Model Configuration

- Architecture: EfficientNetB7
- Input size: 224 × 224 × 3
- Number of classes: 2
- Batch size: 16
- Epochs: 30
- Learning rate: 2e-4
- Mixed precision training enabled

---

## Explainability Methods

### Grad-CAM++
Generates class-discriminative heatmaps highlighting important regions influencing the model’s predictions.

### NAOPC (Normalized Area Over Perturbation Curve)
Evaluates explanation reliability by measuring confidence degradation as important regions are progressively perturbed.

Perturbation strategies:
- Mean replacement
- Zero masking
- Gaussian blur

Interpretation:
- NAOPC > 0.5 : Excellent explanation
- NAOPC 0.3–0.5 : Good explanation
- NAOPC < 0.3 : Weak explanation

---

## Evaluation Metrics

- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro)
- Confusion matrix
- Train vs test performance comparison

---

## Project Structure

