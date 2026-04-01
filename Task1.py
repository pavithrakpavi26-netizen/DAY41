# Step 1: Import libraries
import time
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# Step 2: Load dataset (MNIST: 784 features)
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

# Normalize data
X = X / 255.0

# Use smaller subset for faster execution
X = X[:10000]
y = y[:10000]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

start_time = time.time()

model_raw = LogisticRegression(max_iter=1000)
model_raw.fit(X_train, y_train)

end_time = time.time()

time_raw = end_time - start_time

# Prediction
y_pred_raw = model_raw.predict(X_test)
acc_raw = accuracy_score(y_test, y_pred_raw)

print("WITHOUT PCA")
print("Accuracy:", acc_raw)
print("Training Time:", time_raw)

# Step B: Apply PCA (784 → 50)

pca = PCA(n_components=50)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train again
start_time = time.time()

model_pca = LogisticRegression(max_iter=1000)
model_pca.fit(X_train_pca, y_train)

end_time = time.time()

time_pca = end_time - start_time

# Prediction
y_pred_pca = model_pca.predict(X_test_pca)
acc_pca = accuracy_score(y_test, y_pred_pca)

print("\nWITH PCA")
print("Accuracy:", acc_pca)
print("Training Time:", time_pca)



# Step C: Speedup Factor

speedup = time_raw / time_pca

print("\nSpeedup Factor:", speedup)