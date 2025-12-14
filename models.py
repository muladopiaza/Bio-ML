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
