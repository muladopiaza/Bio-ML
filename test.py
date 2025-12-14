import numpy as np
from models import LogisticRegression
from preprocessing import StandardScaler
from metrics import evaluate
from utils import Pipeline  # your Pipeline class

# 1. Create dataset
X = np.array([[0, 0],
              [1, 0],
              [0, 1],
              [1, 1]])
y = np.array([0, 0, 0, 1])  # OR-like labels

# 2. Create pipeline: scaler + logistic regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(learning_rate=0.1, n_iters=1000))
])

# 3. Fit the pipeline
pipeline.fit(X, y)

# 4. Predict on training data
y_pred = pipeline.predict(X)
y_prob = pipeline.steps[-1][1].predict_proba(pipeline.steps[0][1].transform(X))

# 5. Print results
print("Predicted probabilities:", y_prob)
print("Predicted classes:", y_pred)

# 6. Evaluate
acc = pipeline.score(X, y, metric='accuracy')
loss = pipeline.steps[-1][1]._loss(y, y_prob)
print("Accuracy on training data:", acc)
print("Log-loss on training data:", loss)
