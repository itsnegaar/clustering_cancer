import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA 
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

#load and clear the data
data = pd.read_csv('/home/veunex/a/MachineLearning/clustering/archive/data.csv')
print(data.isnull().any())
# print(data.info)
# print(data.shape)
# print(data.columns)

data.diagnosis = data.diagnosis.replace(['M','B'],[1,0])
data.drop(['diagnosis'] , axis=1)

# print(data.columns)
# plt.figure(figsize = (10, 10))
corr_relation=data.corr()
sns.heatmap(corr_relation,annot=True,cmap="Blues")
plt.show()

plt.figure(figsize = (10, 10))
plt.scatter(data["radius_mean"], data["texture_mean"])
plt.xlabel('radius_mean')
plt.ylabel('texture_mean')
plt.show()


# K_mean algorithm

#choose k_mean - elbow method
wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 10), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

k = 2 

#reduse dimension
pca = PCA(n_components = None) 
data = pca.fit_transform(data) 
explained_variance = pca.explained_variance_ratio_


#fit and predict
kmeans = KMeans(n_clusters = 2, init = 'k-means++')
y_kmeans = kmeans.fit_predict(data)

# print(type(data))
# plt.figure(figsize = (15, 10))
# plt.scatter(data.radius_mean, data.texture_mean, c = y_kmeans, alpha = 0.5)
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color = "red", alpha = 1)
# plt.xlabel('radius_mean')
# plt.ylabel('texture_mean')
# plt.show()

# print(data['perimeter_mean'])
# print(data['area_mean'])
# Visualising the clusters
# plt.scatter(data['perimeter_mean', 0], data['area_mean', 1], s = 100, c = 'blue', label = 'Cluster 1')
# plt.scatter(data['perimeter_mean', 0], data['area_mean'][y_kmeans == 1, 1], s = 100, c = 'red', label = 'Cluster 2')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
# plt.title('Clusters of breast cancer')
# plt.xlabel('PCA 1')
# plt.ylabel('PCA 2')
# plt.legend()
# plt.show()
#cluster

#Agglomerative Clustering
ag = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
y_ag = ag.fit_predict(data)

# Visualising the clusters
plt.scatter(data[y_ag == 0, 0], data[y_ag == 0, 1], s = 100, c = 'yellow', label = 'Cluster 1')
plt.scatter(data[y_ag == 1, 0], data[y_ag == 1, 1], s = 100, c = 'red', label = 'Cluster 2')
plt.title('Clusters of breast cancer')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.show()

# Density-Based Methods
db = DBSCAN(eps=3, min_samples=2).fit(data)
y_db = db.fit_predict(data)

plt.scatter(data['perimeter_mean', 0], data['area_mean', 1][y_db == 1, 1], s = 100, c = 'blue', label = 'Cluster 1')
plt.scatter(data['perimeter_mean', 0], data['area_mean'][y_db == 1, 1], s = 100, c = 'red', label = 'Cluster 2')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of breast cancer')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.show()