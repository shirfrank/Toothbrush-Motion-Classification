from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import numpy as np

def feature_correlation(X_features, Y_vector):
    """
    Computes the mutual information (MI) between a single feature vector and a label vector.

    Parameters:
        X_features (array-like): A 1D array of feature values (can include NaNs).
        Y_vector (array-like): A 1D or 2D array of labels corresponding to each feature value.

    Returns:
        float: The mutual information score between the feature and the label vector.
               Returns 0.0 if not enough valid data is available.
    """
    # Ensure input is a 2D array with one feature column
    X_features = np.array(X_features).reshape(-1, 1)
    Y_vector = np.array(Y_vector)

    # If Y_vector is 2D (e.g., one-hot), reduce it to 1D
    if len(Y_vector.shape) > 1:
        Y_vector = Y_vector[:, 0]

    # Create a mask to remove NaN values from both feature and label arrays
    mask = ~np.isnan(X_features).flatten()
    X_filtered = X_features[mask]
    y_filtered = Y_vector[mask]

    # Not enough data left after removing NaNs
    if len(X_filtered) < 2:
        print("⚠️ Not enough valid values.")
        return 0.0

    # Convert labels to numeric format if categorical
    y_encoded = LabelEncoder().fit_transform(y_filtered)

    # Compute mutual information between feature and encoded label
    mi = mutual_info_classif(X_filtered, y_encoded, discrete_features=False)

    # Return the MI score for the single feature
    return mi[0]
