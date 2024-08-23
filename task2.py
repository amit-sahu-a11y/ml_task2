# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Step 2: Load Data
data = pd.read_csv('Mall_Customers.csv')

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Step 3: Preprocess Data
# Assuming 'Annual Income (k$)' and 'Spending Score (1-100)' are features
features = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 4: Apply K-means Clustering
# Find the optimal number of clusters using the Elbow Method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Fit K-means with the optimal number of clusters (e.g., 5)
k_optimal = 5
kmeans = KMeans(n_clusters=k_optimal, random_state=0)
clusters = kmeans.fit_predict(scaled_features)

# Step 5: Analyze Clusters
data['Cluster'] = clusters
print(data.groupby('Cluster').mean())

# Step 6: Visualize Clusters
plt.figure(figsize=(10, 7))
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data['Cluster'], cmap='viridis')
plt.title('Customer Clusters')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.colorbar(label='Cluster')
plt.show()
