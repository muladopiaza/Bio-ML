import numpy as np

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def evaluate(y_true, y_pred, metric='accuracy'):
    if metric == 'accuracy':
        return accuracy(y_true, y_pred)
    elif metric == 'mse':
        return mse(y_true, y_pred)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
