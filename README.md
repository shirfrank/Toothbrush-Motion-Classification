# Toothbrush Movement Classification using IMU Sensors

## Project Overview

This project focuses on the classification of toothbrush movements using time-series data from inertial sensors (accelerometer, gyroscope, and magnetometer). The goal is to accurately segment, extract features, and classify brushing behaviors using machine learning, specifically a tuned **XGBoost** classifier.

The system includes a full pipeline: from raw CSV data to trained model deployment and final prediction generation for blind test data.

---

## Project Structure

| File / Folder                            | Purpose                                                                      |
| ---------------------------------------- | ---------------------------------------------------------------------------- |
| `main_01.py`                             | Full training pipeline: segmentation, feature extraction, CV, tuning         |
| `test_final.py`                          | Inference script for blind data. Loads trained model and outputs predictions |
| `extract_features.py`                    | Feature extraction from segmented signals                                    |
| `segment_signal.py`                      | Signal segmentation logic for labeled data                                   |
| `split_data.py`                          | Logic for splitting data into training and test sets                         |
| `select_features.py`                     | Feature selection using MRMR + SFS                                           |
| `Vetting_Pipeline.py`                    | Feature vetting using ReliefF with correlation filtering                     |
| `smooth_prediction.py`                   | Optional post-processing for smoothing predictions                           |
| `compute_lle.py`                         | Largest Lyapunov Exponent (LLE) feature implementation                       |
| `feature_correlation.py`                 | Mutual information computation for single feature relevance                  |
| `final_xgb_model_all_data_bundle.joblib` | Trained XGBoost model with scaler and selected feature names                 |

---

## Main Pipeline (`main_01.py`)

1. **Segmentation**:

   - Segments raw sensor data into 1-second windows.

2. **Feature Extraction**:

   - Extracts a wide set of features (time domain, frequency domain, statistical, LLE, etc).

3. **Preprocessing**:

   - Cleans features with NaN/Inf values.
   - Normalizes features with `StandardScaler`.

4. **Cross-Validation**:

   - Uses Leave-One-Group-Out CV on Groups 01–08.
   - For each fold:
     - Features are vetted via ReliefF + correlation filtering.
     - MRMR + SFS selects the top subset.
     - Optuna tunes both XGBoost hyperparameters and class weights.

5. **Final Model Training**:
   - Features from all folds are unified.
   - Final model is trained on the full data (including Group 02).
   - Saved as a bundled file (`final_xgb_model_all_data_bundle.joblib`).

---

## Inference Script (`test_final.py`)

Used for evaluating **blind test data**.

- Expects all test recordings in `Blind_Data/`.
- Segments and extracts features for each recording.
- Loads the trained model and scaler.
- Applies feature selection and normalization.
- Predicts brushing activity for each segment.
- Merges consecutive predictions into continuous intervals.
- Saves:

  - `01_predictions.csv` – prediction output file
  - `01_expected_accuracy.csv` – expected performance

  **Compliant with official submission format**

---

## Features Used

- **Time domain**: mean, std, max, min, RMS, etc.
- **Frequency domain**: peak frequencies, FFT energy
- **Lyapunov exponent**: measures chaos
- **Statistical**: entropy, skewness, kurtosis
- **Gyro/accel vector norms**, **jerk**, **cosine similarity**

---

## Final Model Output

| Key             | Description                               |
| --------------- | ----------------------------------------- |
| `model`         | Trained XGBoost classifier                |
| `scaler`        | StandardScaler used to normalize features |
| `feature_names` | Names of selected features                |
| `params`        | Tuned hyperparameters from CV             |
| `class_weights` | Average class weights from CV             |

---

## Requirements

- Python 3.8+
- Required packages (install via `pip` or `conda`):
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `xgboost`
  - `optuna`
  - `matplotlib`
  - `joblib`
  - `tqdm`
  - `skrebate` (for ReliefF)

---

## Authors

- Yuval Berkovich
- Shir Frank
