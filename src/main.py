import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

test_file = "processed_test.csv"
train_file = "processed_train.csv"

train = pd.read_csv(f"data/{train_file}", engine="python")
test = pd.read_csv(f"data/{test_file}", engine="python")

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

wcss = []
silhoutte = []
for i in range(2, 21):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(train_scaled)
    wcss.append(kmeans.inertia_)
    kmeans_sil = KMeans(n_clusters=i, init='k-means++', random_state=42)
    labels = kmeans_sil.fit_predict(train_scaled)
    silhoutte.append(silhouette_score(train_scaled, labels))

best_clusters = silhoutte.index(max(silhoutte)) + 1
print("Лучшее кол-во кластеров по методу Silhoutte score")

plt.figure()
plt.title("Elbow method (StandardScaler)")
plt.plot(wcss)
plt.xlabel("Cnt of clusters")
plt.ylabel("WCSS")

plt.figure()
plt.title("Silhoutte score (StandardScaler)")
plt.plot(silhoutte)
plt.xlabel("Cnt of clusters")
plt.ylabel("Silhoutte score")

kmeans = KMeans(n_clusters=best_clusters, init="k-means++", random_state=42)
kmeans.fit(train_scaled)

train_clusters = kmeans.predict(train_scaled)
test_clusters = kmeans.predict(test_scaled)

pca = PCA(n_components=2)
train_2d = pca.fit_transform(train_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(train_2d[:, 0], train_2d[:, 1], c=train_clusters, cmap='viridis', alpha=0.6)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('Кластеры в 2D (StandardScaler + PCA)')
plt.colorbar(label='Кластер')

print("Распределения классов:")
print(f"Train: {Counter(train_clusters)}")
print(f"Test: {Counter(test_clusters)}\n")

print(f"PC1 объясняет {pca.explained_variance_ratio_[0]*100:.1f}% вариации")
print(f"PC2 объясняет {pca.explained_variance_ratio_[1]*100:.1f}% вариации")
print(f"Всего: {sum(pca.explained_variance_ratio_)*100:.1f}%\n")

train_with_clusters = train.copy()
train_with_clusters['cluster'] = train_clusters

for i in range(best_clusters):
    cluster_data = train_with_clusters[train_with_clusters['cluster'] == i]
    print(f"Кластер {i} ({len(cluster_data)} отзывов, {len(cluster_data)/len(train)*100:.1f}%)")
    print(cluster_data[['rating', 'effectiveness', 'sideEffects']].mean())
    print()

plt.show()