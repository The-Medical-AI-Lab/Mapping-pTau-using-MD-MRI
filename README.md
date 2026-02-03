# Mapping Phosphorylated Tau using Multidimensional MRI in Alzheimer's Disease

[cite_start]This repository contains the implementation of a novel imaging framework designed to non-invasively estimate the voxelwise concentration and spatial distribution of **phosphorylated tau (pTau)** in Alzheimer’s Disease (AD)[cite: 1, 13]. 

[cite_start]The project integrates **Multidimensional MRI (MD-MRI)** with supervised machine learning to provide a safe alternative to invasive CSF analysis or ionizing tau-PET imaging[cite: 12, 13, 36].

## 📌 Project Overview

[cite_start]Current clinical diagnosis of AD is strongly linked to pTau accumulation, which reflects neuronal dysfunction[cite: 8]. [cite_start]This framework leverages the rich microstructural information embedded in voxelwise diffusion-relaxation joint distributions (T1-D and T2-D) to predict pTau maps that align with histological ground truth[cite: 14, 16, 17].

### Key Methodology:
* [cite_start]**Feature Engineering**: Vectorized 2D joint distributions of $T_1$, $T_2$, and Mean Diffusivity (MD)[cite: 16].
* [cite_start]**Dimensionality Reduction**: Principal Component Analysis (PCA) with a 95% variance threshold[cite: 21].
* [cite_start]**Optimization**: Hyperparameter selection via **Bayesian Optimization** within a nested cross-validation (5x5 folds) framework[cite: 22].
* [cite_start]**Model Variety**: Implementation and comparison of Linear/Quadratic Regression, SVR, SVM, MLP, and Random Forest[cite: 23, 24].

## 📂 Repository Structure

Based on the core MATLAB implementation (see `image_7129c6.png`):

* [cite_start]`data_generation.m`: Preprocessing and artifact removal using binary masks[cite: 20].
* [cite_start]`pca_visualization_T1D.m` / `pca_visualization_T2D.m`: Scripts for dimensionality reduction analysis[cite: 21].
* `regression_T1D_pca_tuning.m` / `classification_ternary_T2D_pca_tuning.m`: Automated hyperparameter and PCA component tuning.
* [cite_start]`slice_regression_T1D_pca.m`: Voxelwise regression execution scripts[cite: 23].
* `image_regression_T1D_cnn_bo.m`: Experimental CNN-based regression with Bayesian Optimization.
* `ot_binary_classification_T1D.m`: Exploration of Optimal Transport (OT) for distribution-based classification.

## 📊 Performance Highlights

[cite_start]The **Random Forest (RF)** model demonstrated superior stability and accuracy[cite: 25, 29]:

| Task | Data Source | Metric ($R^2$ or Accuracy) | Cohen's Kappa |
| :--- | :--- | :--- | :--- |
| **Regression** | T1D | [cite_start]$0.797 \pm 0.007$ ($R^2$) [cite: 26] | - |
| **2-class Classification** | T1D | [cite_start]$0.924 \pm 0.002$ (Acc) [cite: 27] | [cite_start]$0.803 \pm 0.001$ [cite: 27] |
| **3-class Classification** | T2D | [cite_start]$0.858 \pm 0.005$ (Acc) [cite: 28] | [cite_start]$0.820 \pm 0.006$ [cite: 28] |

[cite_start]Quantitative validation using **Structural Similarity Index (SSIM)** confirmed strong spatial concordance with histology (Mean SSIM: 0.81–0.90)[cite: 32].

## 📜 Citation

If you use this framework or data in your research, please cite our ISMRM abstract:

> **Zhang, H.**, Latimer, C. S., Keene, C. D., Benjamini, D., & Kundu, S. (2025). [cite_start]*Mapping Phosphorylated Tau using Multidimensional MRI in Alzheimer's Disease.* [cite: 1, 2, 5]

---
[cite_start]*This research was conducted at Washington University in St. Louis in collaboration with the University of Washington and the NIH[cite: 3, 4].*
