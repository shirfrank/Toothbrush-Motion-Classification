import numpy as np
import pandas as pd

def split_data_leaveoneout(X_features, Y_vector, X_matrix, test_group='09'):
    """
    Splits data into:
    - Train/Validation folds using Leave-One-Group-Out CV (on all groups except `test_group`)
    - Final test set: samples belonging to `test_group` (never used in training/validation)

    Returns:
        - trainval_features: Features of groups != test_group
        - trainval_labels: Labels of groups != test_group
        - trainval_groups: Group labels of groups != test_group
        - test_features: Features of test_group
        - test_labels: Labels of test_group
        - test_indices: Original indices of test_group
    """
    X_df = pd.DataFrame(X_matrix, columns=['start', 'end', 'hand', 'group', 'recording'])
    X_df['label'] = Y_vector
    X_df['global_index'] = np.arange(len(X_df))

    # Split mask for test group
    is_test = X_df['group'] == test_group
    is_trainval = ~is_test

    # Train/val
    trainval_features = X_features[is_trainval.values]
    trainval_labels = Y_vector[is_trainval.values]
    trainval_groups = X_df.loc[is_trainval, 'group'].values

    # Test
    test_features = X_features[is_test.values]
    test_labels = Y_vector[is_test.values]
    test_indices = X_df.loc[is_test, 'global_index'].values

    return trainval_features, trainval_labels, trainval_groups, test_features, test_labels, test_indices
