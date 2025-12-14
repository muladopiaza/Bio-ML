import numpy as np

from models import LinearRegression
from preprocessing import StandardScaler
from utils import Pipeline
from metrics import evaluate

# -----------------------------
# 1. Create training data
# -----------------------------
X = np.array([
    [1, 2],
    [2, 3],
    [4, 5],
    [3, 6]
])

y = np.array([5, 7, 11, 10])

# -----------------------------
# 2. Create pipeline
# -----------------------------
pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", LinearRegression(fit_intercept=True))
])

# -----------------------------
# 3. Fit pipeline
# -----------------------------
pipeline.fit(X, y)

# -----------------------------
# 4. Inspect learned parameters
# -----------------------------
model = pipeline.steps[-1][1]

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# -----------------------------
# 5. Predict on new data
# -----------------------------
X_new = np.array([
    [5, 5],
    [1, 1]
])

y_pred = pipeline.predict(X_new)
print("Predictions:", y_pred)

# -----------------------------
# 6. Evaluate on training data
# -----------------------------
mse_score = pipeline.score(X, y, metric="mse")
print("MSE on training data:", mse_score)
