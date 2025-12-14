import numpy as np
from models import LogisticRegression
from metrics import evaluate
from preprocessing import StandardScaler

# Dataset
X = np.array([[0, 0],
              [1, 0],
              [0, 1],
              [1, 1]])
y = np.array([0, 0, 0, 1])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and train model
model = LogisticRegression(learning_rate=0.1, n_iters=1000)
model.fit(X_scaled, y)

# Print parameters
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Predictions
y_prob = model.predict_proba(X_scaled)
y_pred = model.predict(X_scaled)
print("Predicted probabilities:", y_prob)
print("Predicted classes:", y_pred)

# Evaluate
acc = evaluate(y, y_pred, metric='accuracy')
loss = model._loss(y, y_prob)
print("Accuracy on training data:", acc)
print("Log-loss on training data:", loss)
