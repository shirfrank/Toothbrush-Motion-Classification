# 🪥 Toothbrush Motion Classification using IMU Data

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Focus-Signal_Processing_%26_ML-green.svg)
![License](https://img.shields.io/badge/License-Academic_Use-lightgrey.svg)

## 📌 Project Overview
This repository contains a comprehensive, end-to-end machine learning pipeline designed to classify toothbrushing movements using **IMU (Inertial Measurement Unit)** sensor data. By analyzing accelerometer, gyroscope, and magnetometer signals, the system identifies four distinct motion classes:

* **Up–Down**
* **Right–Left**
* **Circular Movement**
* **Rest**

The project explores the critical relationship between **segmentation window size** and **feature discriminative power**, implementing robust strategies to prevent data leakage and ensure generalization to unseen subjects.

---

## 🧠 Pipeline Architecture

### 1. Signal Segmentation & Preprocessing
Raw multivariate time-series data is segmented into fixed-size windows to evaluate temporal dependencies:
* **Window Sizes:** 0.5s, 3s, and 10s.
* **Output:** Generates a structured `X_matrix` (metadata) and `Y_vector` (labels) based on ground-truth annotations.

### 2. High-Dimensional Feature Extraction
Extracted over **50 features** per segment to capture both time and frequency domain characteristics:
* **Statistical:** Mean, STD, Variance, Skewness, Kurtosis.
* **Frequency Domain:** FFT-based energy and spectral density.
* **Non-Linear Dynamics:** Implementation of **LLE (Locally Linear Embedding)** as a signature feature.

### 3. Rigorous Evaluation Strategies
To ensure the model generalizes to new users, two splitting protocols were implemented:
* **Within-Group Split:** 80/20 stratified split (testing intra-user consistency).
* **Group-Wise Split:** Leave-Groups-Out approach (testing inter-user generalization).

### 4. Feature Engineering & Vetting
A multi-stage pipeline to reduce dimensionality and combat the "Curse of Dimensionality":
* **Normalization:** Group-aware normalization to prevent data leakage.
* **Redundancy Filter:** Removal of features with **Spearman Correlation > 0.8**.
* **Supervised Selection:** Feature ranking using the **Relief Algorithm** to select the top 20 most informative predictors.

---

## 📁 Project Structure

```text
├── data/                         # Raw IMU signals and annotations
├── main_01.py                    # Entry point: executes full pipeline
├── segment_signal.py             # Signal windowing logic
├── extract_features.py           # Core feature extraction engine
├── compute_frequency_features.py # FFT and spectral analysis
├── compute_lle.py                # LLE feature implementation
├── split_data.py                 # Data partitioning logic
├── Vetting_Pipeline.py           # Normalization and feature selection
├── Train_Model.py                # Model training modules
├── Evaluate_Model.py             # Performance metrics and visualization
├── plots/                        # Generated ROC, AUC, and Confusion Matrices
└── README.md
