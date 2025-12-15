import numpy as np

class StandardScaler:
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.min_ = None
        self.max_ = None
        self.feature_range = feature_range

    def fit(self, X):
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        return self

    def transform(self, X):
        scale = self.feature_range[1] - self.feature_range[0]
        return self.feature_range[0] + (X - self.min_) / (self.max_ - self.min_) * scale

    def fit_transform(self, X):
        return self.fit(X).transform(X)
class OneHotEncoder:
    def __init__(self):
        self.categories_ = None

    def fit(self, X):
        self.categories_ = [np.unique(X[:, i]) for i in range(X.shape[1])]
        return self

    def transform(self, X):
        n_samples = X.shape[0]
        result = []
        for i, cats in enumerate(self.categories_):
            col = np.zeros((n_samples, len(cats)))
            for j, cat in enumerate(cats):
                col[:, j] = (X[:, i] == cat).astype(int)
            result.append(col)
        return np.hstack(result)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
class LabelEncoder:
    def __init__(self):
        self.classes_ = None

    def fit(self, y):
        self.classes_ = np.unique(y)
        return self

    def transform(self, y):
        y_transformed = np.zeros_like(y, dtype=int)
        for idx, cls in enumerate(self.classes_):
            y_transformed[y == cls] = idx
        return y_transformed

    def fit_transform(self, y):
        return self.fit(y).transform(y)
class SimpleImputer:
    def __init__(self, strategy='mean', fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None

    def fit(self, X):
        if self.strategy == 'mean':
            self.statistics_ = np.nanmean(X, axis=0)
        elif self.strategy == 'median':
            self.statistics_ = np.nanmedian(X, axis=0)
        elif self.strategy == 'constant':
            self.statistics_ = np.full(X.shape[1], self.fill_value)
        else:
            raise ValueError("Unknown strategy")
        return self

    def transform(self, X):
        X_new = X.copy()
        inds = np.where(np.isnan(X_new))
        X_new[inds] = np.take(self.statistics_, inds[1])
        return X_new

    def fit_transform(self, X):
        return self.fit(X).transform(X)
