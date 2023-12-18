# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Load and explore customer data
customer_data = pd.read_csv('customers.csv')
customer_data.head()
customer_data.describe()
customer_data.info()

# Check for missing values in the dataset
customer_data.isnull().sum()

# Get the shape of the dataset
cust_shape = customer_data.shape
print(cust_shape)

# Encode 'Gender' using LabelEncoder
label_encoder = LabelEncoder()
customer_data['Gender'] = label_encoder.fit_transform(customer_data['Gender'])

# Calculate correlation matrix and plot heatmap
matrix = customer_data.corr()
sns.heatmap(matrix, annot=True)
plt.show()

# Identify columns with minimum correlation
min_corr_params = matrix.abs().idxmin()
col_param_1, col_param_2 = min_corr_params[1], min_corr_params[0]
print(col_param_1, col_param_2)

# Get the maximum correlation value in the matrix
matrix_max = round(matrix[matrix != 1].max().max(), 2)
print('Matrix Max:', matrix_max)

# Select columns for clustering
customer_data = customer_data[[col_param_1, col_param_2]]

# Apply KMeans clustering
num_clust = 15
wcss_list = []

for i in range(1, num_clust + 1):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(customer_data)
    wcss_list.append(kmeans.inertia_)

# Plot the Elbow Method to determine optimal number of clusters
plt.figure(figsize=(20, 10))
plt.plot(range(1, num_clust + 1), wcss_list, marker='o', color='red')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS Values')
plt.show()

# Choose the number of clusters and fit KMeans
chosen_cluster = 6
kmeans = KMeans(n_clusters=chosen_cluster, init='k-means++', n_init=10, random_state=42)
kmeans.fit(customer_data)

# Get cluster assignments and cluster centers
Y = kmeans.fit_predict(customer_data)
print(kmeans.cluster_centers_)
print(Y)

# Plot the clustered data and centroids
max_centre = np.max(kmeans.cluster_centers_)
X = customer_data.iloc[:, :].values

plt.figure(figsize=(20, 10))
plt.scatter(X[Y == 0, 0], X[Y == 0, 1], c='lime', label='Cluster 1')
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], c='maroon', label='Cluster 2')
plt.scatter(X[Y == 2, 0], X[Y == 2, 1], c='gold', label='Cluster 3')
plt.scatter(X[Y == 3, 0], X[Y == 3, 1], c='violet', label='Cluster 4')
plt.scatter(X[Y == 4, 0], X[Y == 4, 1], c='blue', label='Cluster 5')
plt.scatter(X[Y == 5, 0], X[Y == 5, 1], c='black', label='Cluster 6')

# Plot the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='cyan', label='Centroids')

plt.legend(loc="upper right")
plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()
