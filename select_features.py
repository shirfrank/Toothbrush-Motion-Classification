from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from tqdm import trange
import numpy as np

def mrmr_sfs_early_stop(X, y, num_features=20, tolerance=0.0001, min_features=3, verbose=False):
    """
    Feature selection using MRMR (Minimum Redundancy Maximum Relevance) with Sequential Forward Selection (SFS)
    and early stopping based on diminishing improvement.

    Parameters:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Target vector (classification labels).
        num_features (int): Maximum number of features to select.
        tolerance (float): Minimum average improvement required to continue selecting.
        min_features (int): Minimum number of features to select before early stopping can occur.
        verbose (bool): Whether to print progress and details.

    Returns:
        selected (list): Indices of selected features.
        X_selected (np.ndarray): X matrix reduced to selected features.
    """
    print("Using mutual_info_classif for relevance and mutual_info_regression for redundancy")

    X = np.asarray(X)
    # Set number of neighbors for mutual information estimation (smaller for small datasets)
    n_neighbors = min(20, len(X) // 100)

    selected = []                        # Indices of selected features
    remaining = list(range(X.shape[1])) # Indices of unselected (remaining) features

    # Compute mutual information between each feature and the target (relevance)
    relevance = mutual_info_classif(X, y, discrete_features=False, n_neighbors=n_neighbors)
    prev_avg_score = 0  # Initialize previous average score for early stopping check

    for k in trange(num_features, desc="Selecting Features"):
        scores = []  # Store (MRMR score, feature index) for each candidate

        for i in remaining:
            # Compute redundancy as average MI between candidate i and already selected features
            redundancy = np.mean([
                mutual_info_regression(X[:, [i]], X[:, j], n_neighbors=n_neighbors)[0]
                for j in selected
            ]) if selected else 0  # No redundancy if nothing selected yet

            # MRMR score: relevance / redundancy (add epsilon to avoid division by zero)
            score = relevance[i] if redundancy == 0 else relevance[i] / (redundancy + 1e-6)
            scores.append((score, i))

        # Select feature with highest MRMR score
        best_score, best_feature = max(scores)
        selected.append(best_feature)
        remaining.remove(best_feature)

        # Compute average score across candidates to monitor improvement
        current_avg_score = np.mean([s[0] for s in scores])
        improvement = current_avg_score - prev_avg_score
        prev_avg_score = current_avg_score

        # Optionally print step info
        if verbose:
            print(f"Step {k + 1}: Selected feature {best_feature} (score={best_score:.4f}), "
                  f"avg MI={current_avg_score:.4f}, Δ={improvement:.5f}")

        # Early stopping condition: no significant improvement & minimum features selected
        if k + 1 >= min_features and improvement < tolerance:
            if verbose:
                print(f"🛑 Early stopping at {k + 1} features (Δ < {tolerance})")
            break

    # Return selected feature indices and reduced X
    return selected, X[:, selected]
