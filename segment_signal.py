import os
import numpy as np
import pandas as pd

def segment_signal(file_map, window_size, save_path=None):
    """
    Segments labeled time intervals into fixed-size windows and assigns the most dominant label
    to each window, along with its dominance ratio.

    Returns all segments (even with <80% label dominance) for further filtering downstream.

    Parameters
    ----------
    file_map : dict
        Maps keys like '01_02_L' to paths of sensor files, including label files.
    window_size : float
        Duration (in seconds) for each segment window.
    save_path : str, optional
        If provided, saves X_matrix and Y_vector to disk.

    Returns
    -------
    X_matrix : np.ndarray, shape (n_segments, 5)
        Each row: [start_time, end_time, hand, group, recording]
    Y_vector : list of (label, dominance_ratio) tuples
        Each item is (dominant_label, ratio in window)
    skipped_windows : dict
        Maps key to list of skipped window start times (e.g., due to missing label file)
    """

    # Initialize outputs
    X_matrix = []          # Will store metadata for each window
    Y_vector = []          # Will store dominant label and its ratio per window
    skipped_windows = {}   # Logs which windows were skipped per key

    # Iterate over all recordings
    for key, sensors in sorted(file_map.items()):
        # Skip entries with no label file
        if 'label' not in sensors:
            continue

        # Parse metadata from key
        group, rec_num, hand = key.split('_')
        label_path = sensors['label']

        # Try loading label file for the current recording
        try:
            label_df = pd.read_csv(label_path, usecols=[0, 1, 2])
            label_df.columns = ['Start', 'End', 'Label']
        except Exception as e:
            print(f"❌ Error reading {label_path}: {e}")
            continue

        # Define window start times from 0 to total duration
        total_time = label_df['End'].max()
        starts = np.arange(0, total_time, window_size)
        skipped_windows[key] = []

        # Slide a fixed-size window across the timeline
        for start in starts:
            end = start + window_size

            # Find label intervals that overlap with this window
            overlapping = label_df[(label_df['Start'] < end) & (label_df['End'] > start)]
            if overlapping.empty:
                # No label info in this window — log and skip
                skipped_windows[key].append(start)
                continue

            # Compute how long each label is active within the window
            durations = {}
            for _, row in overlapping.iterrows():
                overlap_start = max(start, row['Start'])
                overlap_end = min(end, row['End'])
                duration = overlap_end - overlap_start
                label = row['Label']
                durations[label] = durations.get(label, 0) + duration

            # Pick the label with the longest active duration (dominant)
            dominant_label, dominant_duration = max(durations.items(), key=lambda x: x[1])
            dominance_ratio = dominant_duration / window_size

            # Save the segment and its dominant label (even if dominance < 0.8)
            X_matrix.append([start, end, hand, group, rec_num])
            Y_vector.append((dominant_label, dominance_ratio))

    # Optionally save the results to disk
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        pd.DataFrame(X_matrix, columns=['start', 'end', 'hand', 'group', 'recording']) \
            .to_csv(os.path.join(save_path, "X_matrix.csv"), index=False)
        pd.DataFrame(Y_vector, columns=['label', 'dominance']) \
            .to_csv(os.path.join(save_path, "Y_vector.csv"), index=False)
        print(f"✅ Saved X_matrix and Y_vector to {save_path}")

    # Return numpy array and results
    return np.array(X_matrix), Y_vector, skipped_windows
