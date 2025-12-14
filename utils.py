from models import LinearRegression
from metrics import evaluate
class ModelFactory:
    @staticmethod
    def create(model_name, **kwargs):
        if model_name == 'LinearRegression':
            return LinearRegression(**kwargs)
        else:
            raise ValueError(f"Unknown model: {model_name}")
class Pipeline:
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y):
        for name, step in self.steps[:-1]:  # all but last
            X = step.fit_transform(X)
        # last step is estimator
        self.steps[-1][1].fit(X, y)
        return self

    def predict(self, X):
        for name, step in self.steps[:-1]:
            X = step.transform(X)
        return self.steps[-1][1].predict(X)
    def score(self, X, y, metric='accuracy'):
        y_pred = self.predict(X)
        return evaluate(y, y_pred, metric)
