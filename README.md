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

## 📊 Performance Highlights

Across all tasks, the **Random Forest (RF)** model demonstrated superior stability and predictive accuracy.

| Task | Data Source | Metric (mean ± std) |
|:----:|:-----------:|:-------------------:|
| **Regression** | T1D | R² = 0.797 ± 0.007 |
| **Binary Classification** | T1D | Accuracy = 0.924 ± 0.002 |
| **Ternary Classification** | T2D | Accuracy = 0.858 ± 0.005 |

All results are reported as mean ± standard deviation across outer folds of the nested cross-validation.

Spatial validation using the **Structural Similarity Index (SSIM)** demonstrated strong agreement between predicted pTau maps and histological ground truth, with mean SSIM values ranging from **0.81 to 0.90** across tasks.

---

## 📜 Publication

> **Zhang, H.**, Latimer, C. S., Keene, C. D., Benjamini, D., & Kundu, S. (2025).  
> *Mapping Phosphorylated Tau using Multidimensional MRI in Alzheimer's Disease.*  
> 2026 International Society for Magnetic Resonance in Medicine (ISMRM).

---

## 🤝 Acknowledgements

This research was conducted at **Washington University in St. Louis**, in collaboration with the **University of Washington (UW)**, **Johns Hopkins University (JHU)**, and the **National Institutes of Health (NIH)**. I gratefully acknowledge **Dr. Dan Benjamini** and **Prof. Shinjini Kundu** for their valuable guidance and insightful discussions throughout this research.


