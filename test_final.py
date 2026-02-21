import os
import numpy as np
import pandas as pd
import joblib
from itertools import groupby
from extract_features import extract_features

# === Segmentation function for unlabeled data ===
def segment_signal_unlabeled(file_map, window_size, save_path=None):
    X_matrix = []
    segments = []

    for key, sensors in sorted(file_map.items()):
        group, rec_num, hand = key.split('_')
        dfs = {}
        max_time = 0

        for sensor_type, path in sensors.items():
            df = pd.read_csv(path)
            if 'elapsed (s)' not in df.columns:
                raise ValueError(f"No 'elapsed (s)' column found in {path}")
            df = df.rename(columns={'elapsed (s)': 'time'})
            dfs[sensor_type] = df
            max_time = max(max_time, df['time'].max())

        starts = np.arange(0, max_time, window_size)
        for start in starts:
            end = start + window_size
            segment_data = {
                sensor_type: df[(df['time'] >= start) & (df['time'] < end)].copy()
                for sensor_type, df in dfs.items()
            }
            X_matrix.append([start, end, hand, group, rec_num])
            segments.append({'key': key, 'window': segment_data})

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        pd.DataFrame(X_matrix, columns=['start', 'end', 'hand', 'group', 'recording']) \
            .to_csv(os.path.join(save_path, "X_matrix_unlabeled.csv"), index=False)

    return np.array(X_matrix), segments

# === Configuration ===
WINDOW_SIZE = 1.0
TEST_FOLDER = "Blind_Data"
GROUP_NUMBER = "01"
EXPECTED_ACCURACY = 0.635

# === Construct file map from test folder ===
# === Detect if there's a nested group directory inside Blind_Data ===
inner_dirs = [d for d in os.listdir(TEST_FOLDER) if os.path.isdir(os.path.join(TEST_FOLDER, d))]
if len(inner_dirs) == 1 and inner_dirs[0].isdigit():
    TEST_FOLDER = os.path.join(TEST_FOLDER, inner_dirs[0])
    print(f"Found nested test folder: {TEST_FOLDER}")


file_map = {}
for fname in os.listdir(TEST_FOLDER):
    if not fname.endswith(".csv"):
        continue
    parts = fname.replace(".csv", "").split("_")
    key = "_".join(parts[:3])
    sensor = parts[3]
    if sensor.lower() not in ['acc', 'gyro']:
        continue
    path = os.path.join(TEST_FOLDER, fname)
    file_map.setdefault(key, {})[sensor] = path

# === Segment the signals ===
X_matrix, test_segments = segment_signal_unlabeled(file_map, WINDOW_SIZE)

# === Load trained model and associated components ===
bundle = joblib.load("final_xgb_model_all_data_bundle.joblib")
model = bundle["model"]
scaler = bundle["scaler"]
selected_features = bundle["feature_names"]
all_feature_names = bundle["all_feature_names"]

# === Extract features for all segments ===
dummy_labels = [0] * len(X_matrix)
X_test_array = extract_features(TEST_FOLDER, X_matrix, dummy_labels)
if X_test_array.shape[0] == 0:
    raise ValueError("No segments were processed. Make sure Blind_Data contains valid Acc and Gyro recordings.")

df_test_features = pd.DataFrame(X_test_array, columns=all_feature_names)

# === Apply preprocessing and make predictions ===
missing_features = [f for f in selected_features if f not in df_test_features.columns]
if missing_features:
    raise ValueError(f"Missing features: {missing_features}")

X_test_all_scaled = scaler.transform(df_test_features)
df_test_scaled = pd.DataFrame(X_test_all_scaled, columns=all_feature_names)
X_test = df_test_scaled[selected_features]
Y_pred_test = model.predict(X_test)

# === Convert predictions to official submission format ===
times = np.floor(X_matrix[:, :2].astype(float)).astype(int)
preds = Y_pred_test.tolist()
merged = []
for label, group in groupby(zip(times, preds), key=lambda x: x[1]):
    group = list(group)
    start = group[0][0][0]
    end = group[-1][0][1]
    merged.append([start, end, int(label)])

# === Save prediction file ===
df_pred = pd.DataFrame(merged, columns=["Start", "End", "Label"])
df_pred.to_csv(f"{GROUP_NUMBER}_predictions.csv", index=False)

# === Save expected accuracy file ===
df_acc = pd.DataFrame([[round(EXPECTED_ACCURACY, 4)]], columns=["Expected_Accuracy"])
df_acc.to_csv(f"{GROUP_NUMBER}_expected_accuracy.csv", index=False)
