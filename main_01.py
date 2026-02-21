"""
Main pipeline script for classifying motion segments using sensor data.

This script performs the full machine learning pipeline for time-series classification
based on sensor signals (e.g., accelerometer, gyroscope, magnetometer). The main steps include:

1. Loading and organizing CSV sensor files based on group, recording, hand, and sensor type.
2. Segmenting multi-sensor time-series signals into fixed-size windows (default: 1 second).
3. Extracting a comprehensive set of features per window (including time-domain, frequency-domain, entropy, and LLE features).
4. Cleaning the feature matrix (removing NaN/inf values and invalid rows/columns).
5. Splitting the dataset using Leave-One-Group-Out strategy:
   - Group '02' is held out as the final test set.
   - Groups '01', ..., '08' are used for cross-validation.
6. Performing Leave-One-Group-Out Cross Validation:
   - Normalizing features within each fold.
   - Vetting features using ReliefF.
   - Selecting informative features via MRMR + SFS (Sequential Forward Selection).
   - Tuning XGBoost hyperparameters using Optuna with weighted classes.
   - Training and evaluating each fold, and caching results.
7. Final feature selection:
   - Union of features selected across folds → MRMR + SFS again on full train set.
   - Final feature list saved and reused.
8. Training final model:
   - Averaged hyperparameters and class weights used to train on Groups 01–08.
   - Performance is evaluated on held-out Group 02 using accuracy, ROC AUC, recall, precision, F1, and confusion matrix.
9. Retraining on all available data (Groups 01–09):
   - Final model is trained using all data and saved with its scaler and feature names.

Outputs:
- `X_features.csv`: cached extracted features.
- `fold_cache/`: contains per-fold results with feature selection and tuned hyperparameters.
- `selected_features_final.csv`: names and indices of final selected features.
- `final_xgb_model.joblib`: model trained on training folds only.
- `final_xgb_model_all_data.joblib`: model trained on full dataset.
- `final_xgb_model_all_data_bundle.joblib`: includes model, scaler, params, and feature metadata.
- `plots/confusion_matrix_test.png`: confusion matrix for Group 02.

Dependencies:
- Python packages: numpy, pandas, scikit-learn, xgboost, matplotlib, optuna, joblib
- Local modules: `segment_signal.py`, `extract_features.py`, `split_data.py`, `Vetting_Pipeline.py`, `select_features.py`

Author: Yuval & Shir
"""


import os
import re
import matplotlib
import pandas as pd
import joblib
import numpy as np
matplotlib.use('Agg')  # Disable GUI backend for matplotlib (useful for headless servers)
from segment_signal import segment_signal
from extract_features import extract_features
from xgboost import XGBClassifier
import optuna
from sklearn.preprocessing import label_binarize
from optuna.samplers import TPESampler
from split_data import split_data_leaveoneout
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from Vetting_Pipeline import vet_features, normalize_data
from select_features import mrmr_sfs_early_stop
from smooth_prediction import smooth_predictions
from sklearn.preprocessing import LabelEncoder

# === Settings ===
DATA_PATH = "data"
SAVE_PATH = "."  # Save results in the current project folder
FOLD_CACHE_DIR = "fold_cache"
os.makedirs(FOLD_CACHE_DIR, exist_ok=True)
FINAL_SELECTION_CACHE_FILE = "final_relief_sfs_results.joblib"

# === Step 1: Organize input CSV files ===
def extract_rec_info(filename):
    """
    Extracts metadata from a filename using a regex pattern.
    Returns group, recording number, hand, and sensor type.
    """
    match = re.search(r'(\d{2})_(\d{2})_([LR])_([A-Za-z]+)\.csv$', filename)
    return match.groups() if match else (None, None, None, None)

file_map = {}
for root, _, files in os.walk(DATA_PATH):
    for fname in files:
        if fname.endswith('.csv'):
            group, rec_num, hand, sensor = extract_rec_info(fname)
            if group:
                key = f"{group}_{rec_num}_{hand}"
                file_map.setdefault(key, {})[sensor.lower()] = os.path.join(root, fname)
            else:
                print(f"Skipping unmatched file: {fname}")

# === Step 2: Set fixed window size ===
WINDOW_SIZE = 1  # Set a fixed window size of 1 second
print(f"\nUsing fixed window size: {WINDOW_SIZE} seconds for segmentation and feature extraction")

# === Step 3: Segment signals (keep all segments even if class dominance < 80%)
X_matrix, Y_vector_raw, skipped_windows = segment_signal(
    file_map, window_size=WINDOW_SIZE, save_path=SAVE_PATH)

# Separate labels and dominance ratios
Y_vector = np.array([lbl for lbl, frac in Y_vector_raw])
dominance_ratios = np.array([frac for lbl, frac in Y_vector_raw])

# === Step 4: Load or extract features ===
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
X_FEATURES_PATH = os.path.join(PROJECT_DIR, "X_features.csv")

if os.path.exists(X_FEATURES_PATH):
    print(f"Loading pre-computed features from {X_FEATURES_PATH}")
    X_features = pd.read_csv(X_FEATURES_PATH).values
    feature_names = list(pd.read_csv(X_FEATURES_PATH, nrows=0).columns)
    num_samples = X_features.shape[0]
    X_matrix = X_matrix[:num_samples]
    dominance_ratios = dominance_ratios[:num_samples]
    Y_vector = Y_vector[:num_samples]
else:
    print("Extracting features...")
    X_features = extract_features(DATA_PATH, X_matrix, Y_vector, save_path=PROJECT_DIR)
    feature_names = list(pd.read_csv(X_FEATURES_PATH, nrows=0).columns)

# === Step 5: Clean up problematic feature values (inf, nan, etc.)
X_features_df = pd.DataFrame(X_features, columns=feature_names)

# Replace infinite values with NaN
X_features_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Identify features with any invalid values
invalid_feature_mask = X_features_df.isnull().any()
invalid_features = X_features_df.columns[invalid_feature_mask].tolist()

print(f"Dropping {len(invalid_features)} features with invalid values:")
print(invalid_features)

# Drop invalid columns
X_features_df.drop(columns=invalid_features, inplace=True)
feature_names = list(X_features_df.columns)

# Drop rows with NaN values and update feature matrix
X_features = X_features_df.dropna().values

# Synchronize all arrays to the valid rows only
valid_rows_mask = ~X_features_df.isnull().any(axis=1)
dominance_ratios = dominance_ratios[valid_rows_mask]
X_matrix = X_matrix[valid_rows_mask]
Y_vector = Y_vector[valid_rows_mask]

# === Step 6: Split the dataset ===
# Encode labels as integers
le = LabelEncoder()
Y_vector = le.fit_transform(Y_vector)

# Use Group 02 as the held-out test set
X_trainval, Y_trainval, groups_trainval, X_test, Y_test, test_indices = split_data_leaveoneout(
    X_features, Y_vector, X_matrix, test_group='02'
)

# Debug prints to verify data groupings
print(np.unique(X_matrix[:, 3]))
print(type(X_matrix[0, 3]))
print("Groups in training:", np.unique(groups_trainval))




# Filter out segments in the training+validation set with dominance ratio below 0.8
is_not_test_group = X_matrix[:, 3] != '02'  # Exclude the held-out test group
dominant_mask = dominance_ratios >= 0.8  # Only include dominant class segments
trainval_mask = dominant_mask & is_not_test_group  # Apply both filters

# Apply the mask to create the training+validation data
X_trainval = X_features[trainval_mask]
Y_trainval = Y_vector[trainval_mask]
groups_trainval = X_matrix[trainval_mask, 3]

# === Step 6: Leave-One-Group-Out Cross Validation (on groups 01–08) ===
logo = LeaveOneGroupOut()

# Initialize result containers
selected_feature_sets = []
fold_accuracies = []
fold_roc_aucs = []
fold_recalls = []
fold_precisions = []
fold_f1s = []
all_best_params = []

print("\nStarting Leave-One-Group-Out Cross Validation:")

for fold_idx, (train_idx, val_idx) in enumerate(logo.split(X_trainval, Y_trainval, groups_trainval)):
    print(f"\nFold {fold_idx + 1}:")
    fold_cache_file = os.path.join(FOLD_CACHE_DIR, f"fold_{fold_idx + 1}_results.joblib")

    if os.path.exists(fold_cache_file):
        # Load previously cached results to avoid recomputation
        print(f"Loading cached results for Fold {fold_idx + 1}")
        fold_data = joblib.load(fold_cache_file)
        selected = fold_data["selected"]
        best_params = fold_data["best_params"]
        acc = fold_data["acc"]
        roc = fold_data["roc"]
        recall = fold_data["recall"]
        precision = fold_data["precision"]
        f1 = fold_data["f1"]

    else:
        # Run fold from scratch
        print(f"Running Fold {fold_idx + 1} from scratch...")

        # Split training and validation data for this fold
        X_tr, X_val = X_trainval[train_idx], X_trainval[val_idx]
        Y_tr, Y_val = Y_trainval[train_idx], Y_trainval[val_idx]

        # Normalize training and validation features
        X_tr_norm, X_val_norm = normalize_data(X_tr, X_val)

        # Vet the features using feature importance or correlation, and reduce dimensionality
        X_tr_vetted, X_val_vetted, vetted_names = vet_features(
            X_tr_norm, Y_tr,
            split_type=f'logo_fold_{fold_idx + 1}',
            X_test=X_val_norm,
            k=20,
            feature_names=feature_names
        )

        # Select informative and non-redundant features using MRMR + SFS
        selected, X_tr_sel = mrmr_sfs_early_stop(
            X_tr_vetted, Y_tr,
            num_features=20,
            verbose=False
        )

        X_val_sel = X_val_vetted[:, selected]
        print("\nSelected Features (names):")
        for idx in selected:
            print(f"   - {vetted_names[idx]}")

        # === Define Optuna objective function for hyperparameter tuning ===
        def objective(trial):
            # Define search space for XGBoost hyperparameters
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "eval_metric": "mlogloss",
                "random_state": 42
            }

            # Class weights are optimized separately
            weight_1 = trial.suggest_float("class_weight_1", 1.0, 4.0)
            weight_2 = trial.suggest_float("class_weight_2", 1.0, 4.0)
            weight_3 = trial.suggest_float("class_weight_3", 1.0, 4.0)

            # Build the class weights dictionary
            class_weights = {
                0: 1.0,  # baseline class
                1: weight_1,
                2: weight_2,
                3: weight_3
            }

            # Create training sample weights based on class
            train_weights = np.array([class_weights[y] for y in Y_tr])

            # Train model with current trial's parameters and weights
            clf = XGBClassifier(**params)
            clf.fit(X_tr_sel, Y_tr, sample_weight=train_weights)

            y_pred = clf.predict(X_val_sel)
            return accuracy_score(Y_val, y_pred)

        # Run Optuna to find best hyperparameters
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=100, show_progress_bar=False)

        best_params = study.best_params
        best_class_weights = {
            0: 1.0,
            1: best_params["class_weight_1"],
            2: best_params["class_weight_2"],
            3: best_params["class_weight_3"]
        }

        # Train final model with best parameters
        xgb_params = {k: v for k, v in best_params.items() if not k.startswith("class_weight")}
        model = XGBClassifier(**xgb_params, eval_metric='mlogloss', random_state=42)

        train_weights = np.array([best_class_weights[y] for y in Y_tr])
        model.fit(X_tr_sel, Y_tr, sample_weight=train_weights)

        # Evaluate model on validation set
        Y_pred = model.predict(X_val_sel)

        acc = accuracy_score(Y_val, Y_pred)
        roc = roc_auc_score(label_binarize(Y_val, classes=np.unique(Y_trainval)), model.predict_proba(X_val_sel), multi_class='ovr')
        recall = recall_score(Y_val, Y_pred, average='macro')
        precision = precision_score(Y_val, Y_pred, average='macro')
        f1 = f1_score(Y_val, Y_pred, average='macro')

        # Save results to cache for reuse
        joblib.dump({
            "selected": selected,
            "selected_names": [vetted_names[i] for i in selected],
            "best_params": best_params,
            "class_weights": best_class_weights,
            "acc": acc,
            "roc": roc,
            "recall": recall,
            "precision": precision,
            "f1": f1
        }, fold_cache_file)

        print(f"Saved results for Fold {fold_idx + 1}")

    # Print performance metrics for the current fold
    print(f"Fold Accuracy: {acc:.3f} | ROC AUC: {roc:.3f} | Recall: {recall:.3f} | Precision: {precision:.3f} | F1: {f1:.3f}")

    # Retrieve selected feature names from cache or current run
    try:
        selected_names = fold_data["selected_names"]
        print("Using cached selected feature names")
    except NameError:
        selected_names = [vetted_names[i] for i in selected]
        print("Using selected names from current fold")

    # Store current fold's results
    selected_feature_sets.append(set(selected_names))
    fold_accuracies.append(acc)
    fold_roc_aucs.append(roc)
    fold_recalls.append(recall)
    fold_precisions.append(precision)
    fold_f1s.append(f1)
    all_best_params.append(best_params)


# === Step 7: Final Feature Selection and Training ===
from collections import Counter

# Compute the average class weights across all cross-validation folds
def average_class_weights(fold_dir="fold_cache", num_folds=8):
    weights_list = []
    for i in range(1, num_folds + 1):
        path = os.path.join(fold_dir, f"fold_{i}_results.joblib")
        if os.path.exists(path):
            fold_data = joblib.load(path)
            weights_list.append(fold_data["class_weights"])
    avg_weights = {}
    for cls in weights_list[0].keys():
        avg_weights[cls] = np.mean([w[cls] for w in weights_list])
    return avg_weights

# Compute the average hyperparameters across all folds
def average_hyperparams(fold_dir="fold_cache", num_folds=8):
    all_params = []
    for i in range(1, num_folds + 1):
        path = os.path.join(fold_dir, f"fold_{i}_results.joblib")
        if os.path.exists(path):
            fold_data = joblib.load(path)
            all_params.append(fold_data['best_params'])

    averaged_params = {}
    keys = all_params[0].keys()
    for key in keys:
        values = [params[key] for params in all_params]
        if isinstance(values[0], (int, float)):
            averaged_params[key] = type(values[0])(np.mean(values))
        else:
            averaged_params[key] = Counter(values).most_common(1)[0][0]
    return averaged_params

# Load cached final feature selection if it exists
if os.path.exists(FINAL_SELECTION_CACHE_FILE):
    print("Loading final feature selection from cache...")
    final_data = joblib.load(FINAL_SELECTION_CACHE_FILE)
    X_train_final = final_data["X_train_final"]
    X_test_final = final_data["X_test_final"]
    top_feature_names = final_data["top_feature_names"]

else:
    print("\nRunning final feature selection on full training set...")

    # Compute the union of all selected features across folds (by name)
    union_feature_names = sorted(set().union(*selected_feature_sets))
    print(f"Total unique features selected across folds: {len(union_feature_names)}")

    print("\nFeatures selected across all folds (union):")
    for i, name in enumerate(union_feature_names):
        print(f"{i + 1}. {name}")

    # Convert feature names back to indices
    union_features = [feature_names.index(name) for name in union_feature_names]
    subset_feature_names = union_feature_names

    # Normalize training and test sets
    X_trainval_norm, X_test_norm = normalize_data(X_trainval, X_test)
    X_trainval_subset = X_trainval_norm[:, union_features]
    X_test_subset = X_test_norm[:, union_features]

    # Apply MRMR + SFS on the union subset
    selected_final_refined, _ = mrmr_sfs_early_stop(
        X_trainval_subset,
        Y_trainval,
        num_features=min(20, len(union_features)),
        verbose=True
    )

    print("\nSelected Final Features (names):")
    for idx in selected_final_refined:
        print(f"   - {subset_feature_names[idx]}")

    # Get the top feature names and their indices in the original feature list
    top_feature_names = [subset_feature_names[i] for i in selected_final_refined]
    top_indices = [feature_names.index(name) for name in top_feature_names]

    # Extract only top features after final normalization
    X_trainval_norm, X_test_norm = normalize_data(X_trainval, X_test)
    X_train_final = X_trainval_norm[:, top_indices]
    X_test_final = X_test_norm[:, top_indices]

    # Save selected features to CSV
    df_features = pd.DataFrame({
        'Index in Original Feature List': [feature_names.index(name) for name in top_feature_names],
        'Feature Name': top_feature_names
    })
    df_features.to_csv("selected_features_final.csv", index=False)
    print("Saved selected features to selected_features_final.csv")

    # Cache the final feature matrices and names
    joblib.dump({
        "X_train_final": X_train_final,
        "X_test_final": X_test_final,
        "top_feature_names": top_feature_names
    }, FINAL_SELECTION_CACHE_FILE)
    print("Saved final ReliefF + SFS results to cache.")

# Train final model using averaged hyperparameters from cross-validation
print("\nUsing average hyperparameters from folds to train final model...")
avg_params = average_hyperparams(fold_dir=FOLD_CACHE_DIR, num_folds=8)
print(f"Averaged Params: {avg_params}")

xgb_params = {k: v for k, v in avg_params.items() if not k.startswith("class_weight")}
model = XGBClassifier(**xgb_params, eval_metric='mlogloss', random_state=42)

avg_class_weights = average_class_weights()
final_train_weights = np.array([avg_class_weights[y] for y in Y_trainval])
model.fit(X_train_final, Y_trainval, sample_weight=final_train_weights)

# Predict on training set for evaluation
Y_pred_train = model.predict(X_train_final)

# Compute evaluation metrics on training set
acc_train = accuracy_score(Y_trainval, Y_pred_train)
roc_train = roc_auc_score(label_binarize(Y_trainval, classes=np.unique(Y_trainval)),
                          model.predict_proba(X_train_final), multi_class='ovr')
recall_train = recall_score(Y_trainval, Y_pred_train, average='macro')
precision_train = precision_score(Y_trainval, Y_pred_train, average='macro')
f1_train = f1_score(Y_trainval, Y_pred_train, average='macro')

# Print training performance
print("\nFinal Training Set Scores:")
print(f"Accuracy:  {acc_train:.3f}")
print(f"ROC AUC:   {roc_train:.3f}")
print(f"Recall:    {recall_train:.3f}")
print(f"Precision: {precision_train:.3f}")
print(f"F1 Score:  {f1_train:.3f}")

# Predict on the held-out test set (Group 02)
Y_pred_test = model.predict(X_test_final)

# Compute confusion matrix
cm_test = confusion_matrix(Y_test, Y_pred_test)
print("Final Confusion Matrix (Test Set):")
print(cm_test)

# Display and save confusion matrix plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=le.classes_)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix - Final Test Set (Group 02)")
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/confusion_matrix_test.png")
print("Saved confusion matrix to plots/confusion_matrix_test.png")

# Save the trained model
joblib.dump(model, "final_xgb_model.joblib")
print("Saved trained model to final_xgb_model.joblib")

# Report final accuracy on Group 02
acc = accuracy_score(Y_test, Y_pred_test)
print(f"\nFinal Test Accuracy on Group 02: {acc:.3f}")

# === Final model training using ALL groups (including Group 02) ===

# Step 1: Concatenate all training and test data
X_all = np.concatenate([X_trainval, X_test], axis=0)
Y_all = np.concatenate([Y_trainval, Y_test], axis=0)

# Step 2: Normalize the combined dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_all_norm = scaler.fit_transform(X_all)

# Step 3: Select only the top features
top_indices = [feature_names.index(name) for name in top_feature_names]
X_all_final = X_all_norm[:, top_indices]

# Step 4: Compute final class weights
avg_class_weights = average_class_weights()
final_weights_all = np.array([avg_class_weights[y] for y in Y_all])

# Step 5: Train final model on all available data
xgb_params = {k: v for k, v in avg_params.items() if not k.startswith("class_weight")}
model = XGBClassifier(**xgb_params, eval_metric='mlogloss', random_state=42)
model.fit(X_all_final, Y_all, sample_weight=final_weights_all)

# Save the final model and scaler
joblib.dump(model, "final_xgb_model_all_data.joblib")
joblib.dump(scaler, "final_scaler_all_data.joblib")
print("Trained and saved final model on ALL data")

# Save full model bundle for reuse
joblib.dump({
    "model": model,
    "scaler": scaler,
    "feature_names": top_feature_names,
    "all_feature_names": feature_names,
    "params": avg_params,
    "class_weights": avg_class_weights
}, "final_xgb_model_all_data_bundle.joblib")
print("Saved final model bundle to final_xgb_model_all_data_bundle.joblib")
