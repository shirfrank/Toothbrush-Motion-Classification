import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from skrebate import ReliefF

# Create folder for plots if it doesn't exist
os.makedirs("plots", exist_ok=True)

def normalize_data(X_train, X_test=None):
    """
    Standardizes features by removing the mean and scaling to unit variance (global across all samples).

    Parameters:
        X_train (np.ndarray): Training data
        X_test (np.ndarray): Test data (optional)

    Returns:
        Tuple of (normalized X_train, normalized X_test or None)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    return X_train_scaled, None


def display_correlation_matrix(X, feature_names=None, plot=False, split_type="split"):
    """
    Computes and optionally plots the Spearman correlation matrix of features.

    Parameters:
        X (np.ndarray): Feature matrix
        feature_names (list): List of feature names for labeling (optional)
        plot (bool): Whether to save the correlation heatmap
        split_type (str): Label for saving plot with filename

    Returns:
        corr_matrix (np.ndarray): Spearman correlation matrix
    """
    corr_matrix = pd.DataFrame(X).corr(method='spearman').values
    if plot:
        plt.figure(figsize=(10, 8))
        plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar()
        if feature_names:
            plt.xticks(range(len(feature_names)), feature_names, rotation=90)
            plt.yticks(range(len(feature_names)), feature_names)
        plt.title("Feature Correlation Matrix (Spearman)")
        plt.tight_layout()
        plt.savefig(f"plots/correlation_matrix_{split_type}.png")
        plt.close()

    return corr_matrix


def plot_feature_ranking(relief_model, split_type, feature_names=None, top_k=20):
    """
    Plots and saves the top-k features ranked by ReliefF score.

    Parameters:
        relief_model (ReliefF): Trained ReliefF model
        split_type (str): Label for saving plot
        feature_names (list): Feature names (optional)
        top_k (int): Number of top features to show

    Returns:
        sorted_indices (list): Indices of the top-k ranked features
    """
    scores = relief_model.feature_importances_
    sorted_indices = np.argsort(scores)[::-1][:top_k]
    top_scores = scores[sorted_indices]
    top_names = [feature_names[i] if feature_names else f"F{i}" for i in sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.barh(top_names[::-1], top_scores[::-1])
    plt.xlabel("ReliefF Score")
    plt.title(f"Top {top_k} Features by ReliefF ({split_type.replace('_', ' ').title()})")
    plt.tight_layout()
    plt.savefig(f"plots/feature_ranking_{split_type}.png")
    plt.close()

    return sorted_indices


def vet_features(X_train, Y_train, split_type, X_test=None, k=20, plot_corr=False, feature_names=None):
    """
    Filters and selects the top-k most relevant features using ReliefF with correlation-based filtering.

    Steps:
        1. Compute ReliefF scores on a stratified sample of the training data.
        2. Compute a Spearman correlation matrix.
        3. Identify highly correlated feature pairs (|r| > 0.8).
        4. Remove the less relevant feature from each pair.
        5. Re-train ReliefF on the reduced feature set.
        6. Return the top-k features after filtering.

    Parameters:
        X_train (np.ndarray): Training feature matrix
        Y_train (np.ndarray): Training labels
        split_type (str): Label for saving plots and metadata
        X_test (np.ndarray): Optional test feature matrix
        k (int): Number of features to select after filtering
        plot_corr (bool): Whether to save a plot of the correlation matrix
        feature_names (list): List of original feature names (optional)

    Returns:
        X_train_vetted (np.ndarray): Filtered and selected features for training
        X_test_filtered (np.ndarray): Filtered and selected features for test (if provided)
        selected_feature_names (list): Names of selected features
    """
    # Step 1: Compute initial ReliefF scores (on sampled data if needed)
    n_neighbors = min(20, max(5, len(Y_train) // 10))
    full_relief = ReliefF(n_neighbors=n_neighbors, n_features_to_select=X_train.shape[1])
    MAX_RELIEF_SAMPLES = 10000

    if X_train.shape[0] > MAX_RELIEF_SAMPLES:
        from sklearn.utils import resample
        X_relief_sampled, Y_relief_sampled = resample(
            X_train, Y_train, n_samples=MAX_RELIEF_SAMPLES,
            stratify=Y_train, random_state=42
        )
    else:
        X_relief_sampled, Y_relief_sampled = X_train, Y_train

    full_relief.fit(X_relief_sampled, Y_relief_sampled)
    relief_scores = full_relief.feature_importances_

    # Step 2: Correlation matrix (optional plot)
    corr_matrix = display_correlation_matrix(X_train, feature_names=feature_names, plot=plot_corr, split_type=split_type)

    # Step 3: Identify highly correlated feature pairs (upper triangle only)
    triu_indices = np.triu_indices(corr_matrix.shape[0], k=1)
    corr_values = corr_matrix[triu_indices]
    high_corr_mask = np.abs(corr_values) > 0.8
    high_corr_pairs = list(zip(triu_indices[0][high_corr_mask], triu_indices[1][high_corr_mask]))

    # Step 4: Drop one feature from each correlated pair (the one with lower ReliefF score)
    to_remove = set()
    for i, j in tqdm(high_corr_pairs, desc="📉 Filtering correlated features"):
        if i in to_remove or j in to_remove:
            continue
        if relief_scores[i] < relief_scores[j]:
            to_remove.add(i)
        else:
            to_remove.add(j)

    # Step 5: Report filtered features
    print(f"\n⚠️ Removing {len(to_remove)} highly correlated features (|r| > 0.8) based on ReliefF importance.")
    for i, j in high_corr_pairs:
        if i in to_remove and j not in to_remove:
            print(f"  - Dropping {feature_names[i]} (r={corr_matrix[i, j]:.2f}) over {feature_names[j]}")
        elif j in to_remove and i not in to_remove:
            print(f"  - Dropping {feature_names[j]} (r={corr_matrix[i, j]:.2f}) over {feature_names[i]}")

    # Step 6: Filter feature matrix
    keep_indices = [i for i in range(X_train.shape[1]) if i not in to_remove]
    X_train_reduced = X_train[:, keep_indices]
    X_test_reduced = X_test[:, keep_indices] if X_test is not None else None
    reduced_feature_names = [feature_names[i] for i in keep_indices] if feature_names else None

    # Step 7: Final ReliefF ranking on reduced features
    relief = ReliefF(n_neighbors=n_neighbors, n_features_to_select=k)
    print("⚙️ Running final ReliefF...")

    if X_train_reduced.shape[0] > MAX_RELIEF_SAMPLES:
        X_relief_final, Y_relief_final = resample(
            X_train_reduced, Y_train, n_samples=MAX_RELIEF_SAMPLES,
            stratify=Y_train, random_state=42
        )
    else:
        X_relief_final, Y_relief_final = X_train_reduced, Y_train

    with tqdm(total=1, desc="ReliefF"):
        relief.fit(X_relief_final, Y_relief_final)

    X_train_vetted = relief.transform(X_train_reduced)
    top_k_indices = np.argsort(relief.feature_importances_)[::-1][:k]
    X_test_filtered = X_test_reduced[:, top_k_indices] if X_test_reduced is not None else None
    selected_feature_names = [reduced_feature_names[i] for i in top_k_indices] if reduced_feature_names else None

    print(f"\n✅ Final vetted top-{k} features:")
    for name in selected_feature_names:
        print(f"   - {name}")

    if X_test is not None:
        return X_train_vetted, X_test_filtered, selected_feature_names
    else:
        return X_train_vetted, None, selected_feature_names
