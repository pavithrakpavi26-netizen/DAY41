# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 2: Load dataset
df = pd.read_csv("WineQTcsv")

# Step 3: Separate features and target
X = df.drop('quality', axis=1)   # 11 features
y = df['quality']                # quality labels

# Step 4: Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Apply PCA (11 → 2)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 6: Create scatter plot
plt.figure()

scatter = plt.scatter(X_pca[:, 0],
                      X_pca[:, 1],
                      c=y)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Wine Dataset")

plt.colorbar(scatter, label='Quality')
plt.show()