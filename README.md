# Mapping Phosphorylated Tau using Multidimensional MRI in Alzheimer's Disease

This repository implements a multidimensional MRI (MD-MRI)–based imaging framework for **non-invasive voxelwise estimation of phosphorylated tau (pTau)** concentration and spatial distribution in Alzheimer’s disease (AD).

The proposed framework integrates **voxelwise diffusion–relaxation joint distributions** with **supervised machine learning**, aiming to provide a quantitative and spatially resolved approach for pTau mapping that complements invasive cerebrospinal fluid (CSF) analysis and ionizing tau-PET imaging.

---

## 🧠 Project Overview

Alzheimer’s disease is strongly associated with the abnormal accumulation of phosphorylated tau (pTau), which reflects neuronal dysfunction and neurodegeneration. Conventional pTau assessment methods, such as CSF biomarkers and tau-PET, are either invasive or involve ionizing radiation, limiting their routine applicability.

This project leverages the rich microstructural information encoded in **voxelwise diffusion–relaxation joint distributions** derived from MD-MRI. By learning statistical relationships between these distributions and histology-derived pTau measurements, the framework produces **voxel-level pTau maps** that demonstrate strong spatial concordance with ground-truth pathology.

---

## 🔬 Methodological Pipeline

**Workflow summary**  
Raw MD-MRI → Preprocessing → Joint Distributions → PCA → Supervised Learning → Voxelwise pTau Maps → Histological Validation

### Key Components

- **Feature Representation**  
  Voxelwise 2D joint diffusion–relaxation distributions (T1–D and T2–D) are vectorized to preserve sub-voxel microstructural heterogeneity.

- **Dimensionality Reduction**  
  Principal Component Analysis (PCA) is applied to the high-dimensional distributional features, retaining **95% of total variance** to reduce noise and improve model stability.

- **Modeling Tasks**  
  - Regression: continuous pTau concentration estimation  
  - Binary classification: low vs. high pTau burden  
  - Ternary classification: multi-level pTau stratification  

- **Learning Algorithms**  
  Linear and quadratic regression, support vector machines (linear and RBF), multilayer perceptrons (MLP), and random forest models are implemented and systematically compared.

- **Hyperparameter Optimization**  
  Bayesian Optimization is embedded within a **nested cross-validation framework (5 × 5 folds)** to ensure unbiased performance estimation and robust model selection.

---

## 📂 Repository Structure

The repository is organized around the core MATLAB implementation:

- `data_generation.m`  
  Preprocessing pipeline including artifact removal and binary mask–based voxel selection.

- `pca_visualization_T1D.m`, `pca_visualization_T2D.m`  
  PCA analysis and visualization scripts for diffusion–relaxation feature spaces.

- `regression_T1D_pca_tuning.m`  
  `classification_ternary_T2D_pca_tuning.m`  
  Automated PCA component selection and hyperparameter tuning using Bayesian optimization.

- `slice_regression_T1D_pca.m`  
  Voxelwise regression and reconstruction of spatial pTau maps.

- `image_regression_T1D_cnn_bo.m`  
  Exploratory CNN-based voxelwise regression included for methodological comparison; not the primary modeling approach.

- `ot_binary_classification_T1D.m`  
  Experimental investigation of Optimal Transport–based distances for distribution-level classification.

---

## 📊 Performance Highlights

Across all tasks, the **Random Forest (RF)** model demonstrated superior stability and predictive accuracy.

| Task | Data Source | Metric (mean ± std) | Cohen’s Kappa |
|-----|------------|---------------------|---------------|
| **Regression** | T1D | R² = 0.797 ± 0.007 | – |
| **Binary Classification** | T1D | Accuracy = 0.924 ± 0.002 | 0.803 ± 0.001 |
| **Ternary Classification** | T2D | Accuracy = 0.858 ± 0.005 | 0.820 ± 0.006 |

All results are reported as mean ± standard deviation across outer folds of the nested cross-validation.

Spatial validation using the **Structural Similarity Index (SSIM)** demonstrated strong agreement between predicted pTau maps and histological ground truth, with mean SSIM values ranging from **0.81 to 0.90** across tasks.

---

## 🧪 Notes on Model Design Choices

- **Why PCA instead of end-to-end CNNs?**  
  Given the limited sample size and the distributional nature of MD-MRI features, PCA-based feature extraction improves robustness, interpretability, and generalization compared to fully end-to-end deep learning models.

- **Why Random Forest?**  
  RF models provide strong non-linear modeling capacity, robustness to noise, and intrinsic feature importance measures, making them particularly suitable for high-dimensional biomedical imaging features.

---

## 📜 Citation

If you use this framework, codebase, or methodology in your research, please cite the following ISMRM abstract:

> **Zhang, H.**, Latimer, C. S., Keene, C. D., Benjamini, D., & Kundu, S. (2025).  
> *Mapping Phosphorylated Tau using Multidimensional MRI in Alzheimer's Disease.*  
> Proceedings of the International Society for Magnetic Resonance in Medicine (ISMRM).

---

## 🤝 Acknowledgements

This research was conducted at **Washington University in St. Louis**, in collaboration with the **University of Washington** and the **National Institutes of Health (NIH)**.
