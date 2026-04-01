# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 2: Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Step 3: Select features
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Step 4: Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Apply PCA (all components)
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Step 6: Explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Step 7: Plot Scree Plot (Cumulative)
plt.figure()
plt.plot(range(1, len(cumulative_variance) + 1),
         cumulative_variance,
         marker='o')

plt.axhline(y=0.95, linestyle='--')  # 95% line

plt.title("Cumulative Explained Variance (Scree Plot)")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance")
plt.show()

# Step 8: Find number of components for 95%
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

print("Components needed for 95% variance:", n_components_95)