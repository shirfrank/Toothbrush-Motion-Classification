import numpy as np

def smooth_predictions(preds, window_size=2):
    """
    Smooths a list of predictions by replacing isolated labels surrounded by a different label.

    Parameters:
    - preds (list or np.ndarray): List of predicted labels (strings or ints).
    - window_size (int): Number of neighbors to check on each side (must be ≥1).

    Returns:
    - np.ndarray: Smoothed predictions.
    """
    preds = np.array(preds)
    smoothed = preds.copy()

    for i in range(window_size, len(preds) - window_size):
        before = preds[i - window_size:i]
        after = preds[i + 1:i + 1 + window_size]
        neighbors = np.concatenate([before, after])

        if np.all(neighbors == neighbors[0]) and preds[i] != neighbors[0]:
            smoothed[i] = neighbors[0]

    return smoothed
