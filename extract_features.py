import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch, find_peaks
from compute_lle import compute_lle
from antropy import sample_entropy
from numpy.linalg import norm

def compute_frequency_features(signal, fs=50):
    """
    Extracts frequency-domain features from a 1D signal using Welch's method.
    Features: dominant frequency, spectral centroid, spectral entropy, total energy.
    """
    # Use Welch's method to estimate the power spectral density (PSD) of the signal.
    # Limit the window length to 256 samples or less (if signal is shorter).
    nperseg = min(256, len(signal))

    # If the signal is too short to compute meaningful PSD, return zeros.
    if nperseg < 2:
        return [0.0] * 4

    # Compute frequency bins (freqs) and corresponding PSD values using Welch's method.
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

    # If the signal has no power (flat signal), return zeros.
    if np.sum(psd) == 0:
        return [0.0] * 4

    # 1. Dominant frequency: frequency with the highest power in the PSD.
    dom_freq = freqs[np.argmax(psd)]

    # 2. Spectral centroid: weighted average of frequencies, giving a sense of the "center of mass" of the spectrum.
    centroid = np.sum(freqs * psd) / np.sum(psd)

    # 3. Spectral entropy: a measure of how flat or peaked the spectrum is (higher entropy = more uniform).
    norm_psd = psd / np.sum(psd)  # Normalize the PSD to get a probability distribution.
    spectral_entropy = -np.sum(norm_psd * np.log2(norm_psd + 1e-12))  # Add small constant to avoid log(0)

    # 4. Energy: total power across all frequencies.
    energy = np.sum(psd)

    # Return the four extracted frequency-domain features.
    return [dom_freq, centroid, spectral_entropy, energy]


def zero_crossing_rate(signal):
    """Returns number of times the signal crosses zero."""
    return ((signal[:-1] * signal[1:]) < 0).sum()

def autocorr_peak(signal):
    """
    Computes autocorrelation of the signal and returns:
    - lag of the first peak (excluding zero lag)
    - value of that peak
    """
    corr = np.correlate(signal - np.mean(signal), signal - np.mean(signal), mode='full')
    corr = corr[len(corr)//2:]
    if len(corr) > 1:
        peak_lag = np.argmax(corr[1:]) + 1
        return [peak_lag, corr[peak_lag]]
    return [0, 0]

def cosine_similarity(v1, v2):
    """Returns cosine similarity between two vectors."""
    if norm(v1) == 0 or norm(v2) == 0:
        return 0.0
    return np.dot(v1, v2) / (norm(v1) * norm(v2))

def extract_features(data_path, X_matrix, Y_vector, save_path=None):
    """
    Extracts a rich set of features for each segment from the raw sensor files.
    Processes both accelerometer and gyroscope CSVs per segment.

    Parameters:
        data_path : str
            Root directory of CSV sensor recordings.
        X_matrix : np.ndarray
            Contains [start_time, end_time, hand, group, recording] for each segment.
        Y_vector : list
            Labels corresponding to each segment.
        save_path : str, optional
            Path to save resulting features as CSV.

    Returns:
        np.ndarray
            Feature matrix of shape (n_segments, n_features).
    """
    # === Initialize containers ===
    X_features = []  # Stores feature vectors for all segments
    feature_names = []  # Stores the names of the features
    segment_info = []  # Stores metadata per segment (e.g., start/end time, label)
    valid_sampen_values = []  # Stores valid sample entropy values for fallback use
    sampen_total = 0  # Total attempts to compute sample entropy
    sampen_errors = 0  # Failed attempts to compute sample entropy

    # === Iterate over each segment row in X_matrix ===
    for i, row in enumerate(X_matrix):
        try:
            # Unpack segment metadata
            start_time, end_time, hand, group, rec = float(row[0]), float(row[1]), row[2], row[3], row[4]
            label = int(Y_vector[i])

            # Build path prefix to access sensor files
            prefix = f"{str(group).zfill(2)}_{str(rec).zfill(2)}_{hand}"
            # נסה קודם עם תיקיית group (למקרה של אימון)
            # נסה למצוא את הקבצים בתיקייה פנימית לפי group
            group_folder = os.path.join(data_path, str(group).zfill(2))
            acc_path_nested = os.path.join(group_folder, f"{prefix}_Acc.csv")
            gyro_path_nested = os.path.join(group_folder, f"{prefix}_Gyro.csv")

            # נסה גם לחפש ישירות בתיקיית data_path (למקרה של מבנה שטוח)
            acc_path_flat = os.path.join(data_path, f"{prefix}_Acc.csv")
            gyro_path_flat = os.path.join(data_path, f"{prefix}_Gyro.csv")

            # תעדף את הנתיב הראשון שקיים
            acc_path = acc_path_nested if os.path.exists(acc_path_nested) else acc_path_flat
            gyro_path = gyro_path_nested if os.path.exists(gyro_path_nested) else gyro_path_flat

            # Skip segment if sensor files do not exist
            if not os.path.exists(acc_path) or not os.path.exists(gyro_path):
                continue

            # Load accelerometer and gyroscope data
            acc = pd.read_csv(acc_path)
            gyro = pd.read_csv(gyro_path)
            acc["elapsed (s)"] = acc["elapsed (s)"].astype(float)
            gyro["elapsed (s)"] = gyro["elapsed (s)"].astype(float)

            # Extract only the portion of the signal in the current segment time window
            acc_seg = acc[(acc["elapsed (s)"] >= start_time) & (acc["elapsed (s)"] < end_time)]
            gyro_seg = gyro[(gyro["elapsed (s)"] >= start_time) & (gyro["elapsed (s)"] < end_time)]

            # Skip if the segment is empty
            if acc_seg.empty or gyro_seg.empty:
                continue

            # Organize sensor data by type and axis
            sensor_segments = {
                'acc': acc_seg[['x-axis (g)', 'y-axis (g)', 'z-axis (g)']],
                'gyro': gyro_seg[['x-axis (deg/s)', 'y-axis (deg/s)', 'z-axis (deg/s)']]
            }

            features = []  # Will collect all features for this segment
            names = []  # Names of the features in the same order
            irregularities = []  # Log of any errors or fallback values used

            # === Process each sensor type and axis ===
            for sensor_type, sensor_data in sensor_segments.items():
                for axis in sensor_data.columns:
                    sig = sensor_data[axis].dropna().astype(float).values
                    if len(sig) == 0:
                        continue
                    prefix_feat = f"{sensor_type}_{axis}"

                    # --- Largest Lyapunov Exponent (LLE) ---
                    try:
                        lle = compute_lle(sig)
                        if not np.isfinite(lle):
                            lle = 0.0
                            irregularities.append((f"{prefix_feat}_LLE", "non-finite"))
                    except:
                        lle = 0.0
                        irregularities.append((f"{prefix_feat}_LLE", "exception"))
                    features.append(lle)
                    names.append(f"{prefix_feat}_LLE")

                    # --- Frequency-domain features ---
                    try:
                        freq_features = compute_frequency_features(sig)
                    except:
                        freq_features = [0.0] * 4
                        irregularities.append((f"{prefix_feat}_freq", "error"))
                    features.extend(freq_features)
                    names.extend([
                        f"{prefix_feat}_dom_freq",
                        f"{prefix_feat}_centroid",
                        f"{prefix_feat}_spectral_entropy",
                        f"{prefix_feat}_energy"
                    ])

                    # --- Sample Entropy ---
                    sampen_total += 1
                    try:
                        sampen = sample_entropy(sig)
                        if not np.isfinite(sampen):
                            raise ValueError("non-finite")
                        valid_sampen_values.append(sampen)
                    except:
                        sampen_errors += 1
                        # Fallback: use mean of valid values, or 0.0 if none exist
                        sampen = np.mean(valid_sampen_values) if valid_sampen_values else 0.0
                        irregularities.append((f"{prefix_feat}_sampen", "invalid"))
                    features.append(sampen)
                    names.append(f"{prefix_feat}_sampen")

                    # --- Time-domain statistical features ---
                    stat_features = [
                        np.mean(sig), np.std(sig), np.min(sig), np.max(sig), np.ptp(sig),
                        np.median(sig), np.sum(sig), np.sqrt(np.mean(sig ** 2)),
                        skew(sig), kurtosis(sig), zero_crossing_rate(sig)
                    ]
                    stat_names = [
                        f"{prefix_feat}_mean", f"{prefix_feat}_std", f"{prefix_feat}_min", f"{prefix_feat}_max",
                        f"{prefix_feat}_ptp", f"{prefix_feat}_median", f"{prefix_feat}_sum", f"{prefix_feat}_rms",
                        f"{prefix_feat}_skew", f"{prefix_feat}_kurtosis", f"{prefix_feat}_zcr"
                    ]
                    features.extend(stat_features)
                    names.extend(stat_names)

                    # --- Autocorrelation peak (lag and value) ---
                    try:
                        peak_lag, peak_val = autocorr_peak(sig)
                    except:
                        peak_lag, peak_val = 0, 0
                        irregularities.append((f"{prefix_feat}_autocorr", "error"))
                    features.extend([peak_lag, peak_val])
                    names.extend([f"{prefix_feat}_autocorr_lag", f"{prefix_feat}_autocorr_val"])

                    # --- Jerk (mean abs derivative) ---
                    jerk = np.mean(np.abs(np.diff(sig))) if len(sig) > 1 else 0.0
                    features.append(jerk)
                    names.append(f"{prefix_feat}_jerk")

                    # --- Peak count ---
                    peak_count = len(find_peaks(sig)[0])
                    features.append(peak_count)
                    names.append(f"{prefix_feat}_peakcount")

                # === Geometric / Axis-based features ===
                try:
                    # Extract x, y, z signals for the current sensor (acc or gyro)
                    x, y, z = [sensor_data[c].dropna().astype(float).values for c in sensor_data.columns]
                    # Compute standard deviation along each axis
                    axis_std = [np.std(x), np.std(y), np.std(z)]
                    # Dominant axis: the one with the highest standard deviation (most movement)
                    dominant_axis = np.argmax(axis_std)
                    # Skew ratio between x and y axes (added small constant to avoid division by zero)
                    skew_ratio = (skew(x) + 1e-6) / (skew(y) + 1e-6)
                except:
                    dominant_axis, skew_ratio = 0.0, 0.0
                 # Append axis-based features
                features.extend([dominant_axis, skew_ratio])
                names.extend([f"{sensor_type}_dominant_axis", f"{sensor_type}_skew_ratio_xy"])

                # === Magnitude-based features ===
                try:
                    magnitude = np.sqrt(x**2 + y**2 + z**2)
                    features.extend([np.mean(magnitude), np.std(magnitude)])
                except:
                    features.extend([0.0, 0.0])
                names.extend([f"{sensor_type}_mag_mean", f"{sensor_type}_mag_std"])

                # Cosine similarity between consecutive vectors
                try:
                    xyz = sensor_data.dropna().astype(float).values
                    sim = [cosine_similarity(xyz[j], xyz[j + 1]) for j in range(len(xyz) - 1)]
                    cos_avg = np.mean(sim) if sim else 0.0
                except:
                    cos_avg = 0.0
                    irregularities.append((f"{sensor_type}_cos_sim", "error"))
                features.append(cos_avg)
                names.append(f"{sensor_type}_cos_sim")

            # === Ratio of rotational to linear movement energy ===
            try:
                # Compute energy as sum of squared magnitudes for acc and gyro
                acc_data = sensor_segments['acc'].dropna().astype(float).values
                gyro_data = sensor_segments['gyro'].dropna().astype(float).values
                acc_energy = np.sum(np.linalg.norm(acc_data, axis=1) ** 2)
                gyro_energy = np.sum(np.linalg.norm(gyro_data, axis=1) ** 2)
                # Energy ratio: rotation (gyro) / linear (acc)
                ratio = gyro_energy / (acc_energy + 1e-6)
            except:
                ratio = 0.0
                irregularities.append(("rotation_linear_ratio", "error"))
            features.append(ratio)
            names.append("rotation_linear_ratio")
            # === Save features for current segment ===
            if features:
                X_features.append(features)
                segment_info.append({
                    'segment_index': i,
                    'recording': prefix,
                    'start_time': start_time,
                    'end_time': end_time,
                    'label': label,
                    'irregularities': irregularities
                })
                if not feature_names:
                    feature_names = names
        # === Final Saving & Reporting ===
        except:
            continue

    # Convert features to DataFrame
    df = pd.DataFrame(X_features, columns=feature_names) if X_features else pd.DataFrame()
    if save_path and not df.empty:
        df.to_csv(os.path.join(save_path, "X_features.csv"), index=False)
    if sampen_total > 0:
        percent = (sampen_errors / sampen_total) * 100
        print(f"⚠️ Replaced {sampen_errors} out of {sampen_total} sampen values ({percent:.2f}%)")

    return df.values
