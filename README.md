# PRODIGY_ML_TASK-02

This code implements customer segmentation using the K-Means clustering algorithm, which is a popular unsupervised machine learning technique for grouping similar data points together based on their features. The goal is to group customers of a mall into distinct segments based on two key features: Annual Income and Spending Score.

Data Loading and Preprocessing:

The dataset containing customer information is loaded from a CSV file. The relevant features (annual income and spending score) are selected for clustering.
Elbow Method for Optimal Clusters:

The code uses the Elbow Method to determine the optimal number of clusters for K-Means. It runs the K-Means algorithm for a range of cluster values (from 1 to 10) and calculates the inertia (within-cluster sum of squared distances). The optimal number of clusters corresponds to the point where inertia begins to decrease at a slower rate, forming an "elbow."
K-Means Clustering:

After determining the optimal number of clusters, the K-Means algorithm is applied to the data to assign each customer to one of the clusters based on their income and spending behavior.
Visualization:

The results are visualized using a scatter plot, with different clusters displayed in different colors. The centroids (mean points of each cluster) are marked on the plot to show the center of each group.
Saving Results:

Finally, the original dataset is enhanced with the cluster labels and saved to a new CSV file, which can be used for further analysis or business strategies.
