from sklearn.model_selection import train_test_split
import numpy as np

def split_train_val_test(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=None, stratify=None):
    assert train_ratio+val_ratio+test_ratio == 1.0, "Invalid ratios!"
    assert train_ratio > 0 and val_ratio > 0 and test_ratio > 0, "Use positive ratios!"

    # calculate the following splits in your solution:
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=True, stratify=stratify, random_state=random_state)
    if stratify is None:
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio/(1-test_ratio), shuffle=True, random_state = random_state)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio/(1-test_ratio), shuffle=True, stratify=y_temp, random_state = random_state)

    return X_train, y_train, X_val, y_val, X_test, y_test

def split_train_val_test_indices(y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=None, stratify=None):
    X_dummy = np.arange(len(y))
    train_indices, y_train, val_indices, y_val, test_indices, y_test = split_train_val_test (X_dummy, y, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, random_state=random_state, stratify=stratify)

    return train_indices, y_train, val_indices, y_val, test_indices, y_test


