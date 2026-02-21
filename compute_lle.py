import numpy as np
from scipy.signal import correlate
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

def estimate_delay_cc(signal, max_lag=100):
    """
    Estimate optimal delay (τ) using the C-C method based on autocorrelation.

    Parameters:
        signal (np.ndarray): The input 1D signal.
        max_lag (int): Maximum lag to test for detecting first minimum.

    Returns:
        int: Estimated time delay τ.
    """
    if np.std(signal) == 0:
        return 1  # fallback if the signal is constant

    # Normalize the signal
    signal = (signal - np.mean(signal)) / np.std(signal)

    # Compute autocorrelation
    acf = correlate(signal, signal, mode='full')[len(signal)-1:]
    acf /= acf[0]  # normalize by zero-lag value

    # Find first local minimum of ACF
    for t in range(1, max_lag - 1):
        if acf[t] < acf[t - 1] and acf[t] < acf[t + 1]:
            return t

    return 1  # fallback if no minimum found

def estimate_embedding_gp(signal, max_dim=15):
    """
    Estimate optimal embedding dimension using a simplified Grassberger-Procaccia method.
    Based on the average nearest-neighbor distance dropping below a threshold.

    Parameters:
        signal (np.ndarray): The input 1D signal.
        max_dim (int): Maximum embedding dimension to consider.

    Returns:
        int: Estimated embedding dimension m.
    """
    N = len(signal)

    for m in range(2, max_dim + 1):
        try:
            # Create embedded vectors (delay = 1)
            X = np.array([signal[i:N - (m - 1) + i] for i in range(m)]).T

            if len(X) < 2:
                continue

            # Compute nearest neighbor distances
            nn = NearestNeighbors(n_neighbors=2).fit(X)
            distances, _ = nn.kneighbors(X)
            mean_dist = np.mean(distances[:, 1])

            # Heuristic threshold: if distances drop too low, embedding is sufficient
            if mean_dist < 0.1:
                return m
        except:
            continue

    return 10  # fallback if no good dimension found

def compute_lle(signal, max_lag=100, max_dim=15):
    """
    Compute the Largest Lyapunov Exponent (LLE) for a 1D time-series signal.
    Uses time-delay embedding and nearest-neighbor divergence over time.

    Parameters:
        signal (np.ndarray): Input time-series signal.
        max_lag (int): Max lag for delay estimation via C-C method.
        max_dim (int): Max embedding dimension via G-P method.

    Returns:
        float: Estimated LLE value. np.nan if estimation fails.
    """
    # Step 1: Estimate time delay (τ) and embedding dimension (m)
    tau = estimate_delay_cc(signal, max_lag=max_lag)
    m = estimate_embedding_gp(signal, max_dim=max_dim)
    N = len(signal)

    # Check if signal is long enough for embedding
    if N < (m - 1) * tau:
        return np.nan

    # Step 2: Time-delay embedding
    X = np.array([signal[i:N - (m - 1) * tau + i:tau] for i in range(m)]).T

    # Step 3: Find nearest neighbors for each point
    distances = cdist(X, X)
    np.fill_diagonal(distances, np.inf)  # ignore self-distance
    neighbors = np.argmin(distances, axis=1)

    # Step 4: Compute divergence (log distance growth) over next time step
    divergence = []
    for i in range(min(len(X) - 1, 50)):
        j = neighbors[i]
        if j + 1 >= len(X):
            continue
        try:
            # Compute distance at next time step
            dist = np.linalg.norm(X[i + 1] - X[j + 1]) + 1e-10
            divergence.append(np.log(dist))
        except:
            continue

    if len(divergence) < 2:
        return np.nan  # not enough valid pairs

    # Step 5: Linear fit to log-divergence to estimate LLE
    times = np.arange(1, len(divergence) + 1)
    coeffs = np.polyfit(times, divergence, 1)

    return coeffs[0]  # slope = LLE estimate
