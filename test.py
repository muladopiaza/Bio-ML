import numpy as np
from models import (
    LinearRegression,
    LogisticRegression,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier,
    KMeans,
    LinearSVC
)
from preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, SimpleImputer
from utils import Pipeline
from metrics import evaluate, train_test_split

# ---------------------------
# 1. Create synthetic datasets
# ---------------------------
# Classification dataset
X_class = np.array([[0, 0],
                    [1, 0],
                    [0, 1],
                    [1, 1]])
y_class = np.array([0, 0, 0, 1])  # OR-like labels

# Regression dataset
X_reg = np.array([[1], [2], [3], [4]])
y_reg = np.array([2, 4, 6, 8])  # y = 2*x

# ---------------------------
# 2. Train/test split
# ---------------------------
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_class, y_class, test_size=0.5, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.5, random_state=42)

# ---------------------------
# 3. Transformers test
# ---------------------------
print("Testing Transformers...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train_reg)
print("StandardScaler result:", X_scaled)

minmax = MinMaxScaler()
X_minmax = minmax.fit_transform(X_train_reg)
print("MinMaxScaler result:", X_minmax)

encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(np.array([[0, 1], [1, 0]]))
print("OneHotEncoder result:\n", X_encoded)

label_enc = LabelEncoder()
y_encoded = label_enc.fit_transform(np.array([0, 1, 1, 0]))
print("LabelEncoder result:", y_encoded)

imputer = SimpleImputer(strategy='mean')
X_missing = np.array([[1, np.nan], [3, 4]])
X_imputed = imputer.fit_transform(X_missing)
print("SimpleImputer result:\n", X_imputed)

# ---------------------------
# 4. Models test
# ---------------------------
print("\nTesting Models...")

models_cls = [
    LogisticRegression(learning_rate=0.1, n_iters=500),
    DecisionTreeClassifier(max_depth=3),
    RandomForestClassifier(n_trees=5, max_depth=3),
    GradientBoostingClassifier(n_estimators=3, learning_rate=0.5),
    LinearSVC(learning_rate=0.01, n_iters=500)
]

models_reg = [
    LinearRegression(),
    DecisionTreeRegressor(max_depth=3)
]

# Test classification models
for model in models_cls:
    model_name = model.__class__.__name__
    print(f"\nTraining {model_name}...")
    model.fit(X_train_cls, y_train_cls)
    y_pred_train = model.predict(X_train_cls)
    y_pred_test = model.predict(X_test_cls)
    acc_train = evaluate(y_train_cls, y_pred_train, metric='accuracy')
    acc_test = evaluate(y_test_cls, y_pred_test, metric='accuracy')
    print(f"{model_name} - Train Accuracy: {acc_train}, Test Accuracy: {acc_test}")

# Test regression models
for model in models_reg:
    model_name = model.__class__.__name__
    print(f"\nTraining {model_name}...")
    model.fit(X_train_reg, y_train_reg)
    y_pred_train = model.predict(X_train_reg)
    y_pred_test = model.predict(X_test_reg)
    mse_train = evaluate(y_train_reg, y_pred_train, metric='mse')
    mse_test = evaluate(y_test_reg, y_pred_test, metric='mse')
    print(f"{model_name} - Train MSE: {mse_train}, Test MSE: {mse_test}")

# ---------------------------
# 5. Pipeline test
# ---------------------------
print("\nTesting Pipeline with LogisticRegression and StandardScaler...")

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(learning_rate=0.1, n_iters=500))
])

pipeline.fit(X_train_cls, y_train_cls)
y_pred_pipeline = pipeline.predict(X_test_cls)
acc_pipeline = pipeline.score(X_test_cls, y_test_cls, metric='accuracy')
print("Pipeline Predictions:", y_pred_pipeline)
print("Pipeline Accuracy:", acc_pipeline)
