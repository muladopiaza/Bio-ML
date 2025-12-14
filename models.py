from abc import abstractmethod,ABC
from metrics import evaluate 
import numpy as np 
class BaseModel(ABC):
    def __init__(self):
        self.is_fitted = False
    @abstractmethod
    def fit(self,X,y):
        """Trains the model on the data using X"""
        pass
    @abstractmethod
    def predict(self,X):
        pass
    def score(self, X, y, metric='accuracy'):
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet")
        y_pred = self.predict(X)
        return evaluate(y, y_pred, metric)
    
class LinearRegression(BaseModel):
    def __init__(self,fit_intercept = True):
        super().__init__()
        self.fit_intercept = fit_intercept
        self.coef_ = None # _ at the end is a convention in ML libraries that this is learned during the fit() and not set manually 
        self.intercept_ = None # _ at the end is a convention in ML libraries that this is learned during the fit() and not set manually 
    def fit(self,X,y):
        if self.fit_intercept:
            X_aug = np.hstack([np.ones((X.shape[0], 1)), X]) # adds a column of ones on the left most sid of the matrix and X now becomes aug_X or augmented matrix 
        else:
            X_aug = X
        w = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y # .T is for transpose @ is for matrix multiplication and np.linalg.inv takes the inverse 
        if self.fit_intercept:
            self.intercept_ = w[0]
            self.coef_ = w[1:]
        else:
            self.intercept_=0
            self.coef_ = w
        self.is_fitted = True
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("The model is not fitted")
        y_pred = X @ self.coef_ + self.intercept_
        return y_pred
    
class LogisticRegression(BaseModel):
    def __init__(self,learning_rate=0.01,n_iters=1000,fit_intercept = True,): # this model will only work with gradient descent optimization hence the learning rate and no of interations 

        super().__init__()
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        # the sigmoid is used to wrap around the linear equation so it returns a value of either 0 or 1 basically a probability of wether it is zero or one 

    def _sigmoid(self, z):
        z = np.asarray(z)
        return np.where(
            z >= 0,
            1 / (1 + np.exp(-z)),
            np.exp(z) / (1 + np.exp(z))
    )
    def _loss(self, y, y_pred):
        y = np.asarray(y)
        y_pred = np.asarray(y_pred)

        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)

        loss = -np.mean(
            y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)
        )

        return loss
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape

        # Initialize parameters
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0

        # Gradient descent
        for _ in range(self.n_iters):
            # Linear model
            z = X @ self.coef_ + self.intercept_

            # Sigmoid prediction
            y_hat = self._sigmoid(z)

        # Gradients
            dw = (1 / n_samples) * (X.T @ (y_hat - y))
            db = (1 / n_samples) * np.sum(y_hat - y)

        # Parameter update
            self.coef_ -= self.learning_rate * dw
            self.intercept_ -= self.learning_rate * db

    # Mark as fitted
        self.is_fitted = True
    def predict_proba(self,X):
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet")
        X = np.asarray(X)
        z = X @ self.coef_ + self.intercept_
        y_hat = self._sigmoid(z)
        return y_hat
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet")
        y_prob = self.predict_proba(X)
        return (y_prob >=0.5).astype(int)
    



class DecisionTreeClassifier(BaseModel):
    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def __init__(self, max_depth=5, min_samples_split=2):
        super().__init__()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        self.tree_ = self._build_tree(X, y)
        self.is_fitted = True

    def _build_tree(self, X, y, depth=0):
        # base case: stop splitting
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            value = np.bincount(y).argmax()
            return self.Node(value=value)

        # find best split
        feature, threshold = self._best_split(X, y)
        if feature is None:
            value = np.bincount(y).argmax()
            return self.Node(value=value)

        # split
        left_idx = X[:, feature] <= threshold
        right_idx = X[:, feature] > threshold
        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        return self.Node(feature, threshold, left, right)

    def _best_split(self, X, y):
        # simplified: try all features, pick first threshold that improves Gini
        best_gini = 1.0
        best_feature, best_threshold = None, None
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left_idx = X[:, feature] <= t
                right_idx = X[:, feature] > t
                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue
                gini = self._gini_index(y[left_idx], y[right_idx])
                if gini < best_gini:
                    best_gini = gini
                    best_feature, best_threshold = feature, t
        return best_feature, best_threshold

    def _gini_index(self, left, right):
        n = len(left) + len(right)
        gini = 0.0
        for group in [left, right]:
            size = len(group)
            if size == 0: continue
            score = 0.0
            classes = np.unique(np.concatenate([left, right]))
            for c in classes:
                p = np.sum(group == c) / size
                score += p ** 2
            gini += (1 - score) * size / n
        return gini

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return np.array([self._traverse_tree(x, self.tree_) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
        


class DecisionTreeRegressor(BaseModel):
    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def __init__(self, max_depth=5, min_samples_split=2):
        super().__init__()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.tree_ = self._build_tree(X, y)
        self.is_fitted = True

    def _build_tree(self, X, y, depth=0):
        # stop condition
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return self.Node(value=np.mean(y))

        # find best split
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return self.Node(value=np.mean(y))

        # split
        left_idx = X[:, feature] <= threshold
        right_idx = X[:, feature] > threshold
        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        return self.Node(feature, threshold, left, right)

    def _best_split(self, X, y):
        best_mse = float('inf')
        best_feature, best_threshold = None, None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left_idx = X[:, feature] <= t
                right_idx = X[:, feature] > t
                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue
                mse = self._mse_split(y[left_idx], y[right_idx])
                if mse < best_mse:
                    best_mse = mse
                    best_feature, best_threshold = feature, t
        return best_feature, best_threshold

    def _mse_split(self, left, right):
        n_left = len(left)
        n_right = len(right)
        n_total = n_left + n_right
        mse_left = np.var(left) if n_left > 0 else 0
        mse_right = np.var(right) if n_right > 0 else 0
        return (n_left * mse_left + n_right * mse_right) / n_total

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return np.array([self._traverse_tree(x, self.tree_) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

class RandomForestClassifier(BaseModel):
    def __init__(self, n_trees=10, max_depth=5, min_samples_split=2):
        super().__init__()
        self.n_trees = n_trees
        self.trees = [DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split) for _ in range(n_trees)]

    def fit(self, X, y):
        n_samples = X.shape[0]
        for tree in self.trees:
            idx = np.random.choice(n_samples, n_samples, replace=True)
            tree.fit(X[idx], y[idx])
        self.is_fitted = True

    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        return np.array([np.bincount(col).argmax() for col in preds.T])
class GradientBoostingClassifier(BaseModel):
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.trees = []

    def fit(self, X, y):
        # Initialize prediction with log odds
        y_pred = np.full(y.shape, np.mean(y))
        for _ in range(self.n_estimators):
            residual = y - y_pred
            tree = DecisionTreeRegressor(max_depth=3)
            tree.fit(X, residual)
            update = self.learning_rate * tree.predict(X)
            y_pred += update
            self.trees.append(tree)
        self.is_fitted = True

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return (y_pred > 0.5).astype(int)
class KMeans(BaseModel):
    def __init__(self, n_clusters=3, max_iters=100):
        super().__init__()
        self.n_clusters = n_clusters
        self.max_iters = max_iters

    def fit(self, X):
        n_samples, n_features = X.shape
        self.centroids_ = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        for _ in range(self.max_iters):
            distances = np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=2)
            self.labels_ = np.argmin(distances, axis=1)
            for k in range(self.n_clusters):
                points = X[self.labels_ == k]
                if len(points) > 0:
                    self.centroids_[k] = points.mean(axis=0)
        self.is_fitted = True

    def predict(self, X):
        distances = np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=2)
        return np.argmin(distances, axis=1)
class LinearSVC(BaseModel):
    def __init__(self, learning_rate=0.01, n_iters=1000, C=1.0):
        super().__init__()
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.C = C

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.coef_) + self.intercept_) >= 1
                if condition:
                    self.coef_ -= self.learning_rate * (2 * 1/self.n_iters * self.coef_)
                else:
                    self.coef_ -= self.learning_rate * (2 * 1/self.n_iters * self.coef_ - np.dot(x_i, y_[idx]) * self.C)
                    self.intercept_ -= self.learning_rate * y_[idx] * self.C
        self.is_fitted = True

    def predict(self, X):
        return np.sign(np.dot(X, self.coef_) + self.intercept_).astype(int)
