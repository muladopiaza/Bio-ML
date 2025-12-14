import numpy as np

# Regression metrics


def mse(y_true, y_pred):
    """Mean Squared Error"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(mse(y_true, y_pred))

def mae(y_true, y_pred):
    """Mean Absolute Error"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    """Coefficient of determination"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


# Classification metrics


def accuracy(y_true, y_pred):
    """Accuracy"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    """Precision for binary classification"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return tp / (tp + fp + 1e-15)  # avoid division by zero

def recall(y_true, y_pred):
    """Recall / Sensitivity for binary classification"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp / (tp + fn + 1e-15)

def f1_score(y_true, y_pred):
    """F1 Score"""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-15)


# Unified evaluate function


def evaluate(y_true, y_pred, metric='accuracy'):
    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1_score,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2_score
    }

    if metric not in metrics_dict:
        raise ValueError(f"Unsupported metric: {metric}")

    return metrics_dict[metric](y_true, y_pred)
