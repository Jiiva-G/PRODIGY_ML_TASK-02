import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#load the data
file_path = "D:/ML_Intern_Task/Task-2/Mall_Customers.csv"  # Replace with your file path
data = pd.read_csv("D:/ML_Intern_Task/Task-2/Mall_Customers.csv")
print(data.head())

# Selecting relevant features for clustering
features = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Determining the optimal number of clusters using the Elbow Method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features)
    inertia.append(kmeans.inertia_)

# Plotting the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(range(1, 11))
plt.grid()
plt.show()

# Implementing K-means clustering with an optimal number of clusters (e.g., 5)
optimal_clusters = 5
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans.fit_predict(features)

# Adding the cluster labels to the original data
data['Cluster'] = clusters

# Visualizing the clusters
plt.figure(figsize=(10, 6))
for cluster in range(optimal_clusters):
    cluster_data = features[data['Cluster'] == cluster]
    plt.scatter(cluster_data['Annual Income (k$)'],
                cluster_data['Spending Score (1-100)'],
                label=f'Cluster {cluster}', s=50)

# Marking centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200, label='Centroids')

plt.title('Customer Segmentation using K-Means')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid()
plt.show()

# Save the clustered dataset
output_file_path = "D:/ML_Intern_Task/Task-2/clustered_customers.csv"  # Specify the file name
data.to_csv(output_file_path, index=False)
print(f"Clustered dataset saved to {output_file_path}")