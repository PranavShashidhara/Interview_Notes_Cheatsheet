# Clustering and Unsupervised Learning

## Overview

**Clustering**: partition data into groups (clusters) based on similarity; no labels.

**Challenge**: no ground truth; subjective notion of "good" clustering; choosing k.

**Applications**: customer segmentation, document organization, image compression, anomaly detection.

---

## K-Means

**Objective**: minimize within-cluster sum of squares (WCSS).

```
WCSS = Σ_i Σ_x∈C_i ||x - μ_i||²
```

where μ_i = centroid of cluster i

**Algorithm**:
1. Initialize k centroids randomly
2. Repeat until convergence:
   - Assign each point to nearest centroid
   - Update centroids = mean of assigned points

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_
```

**Pros**: fast, simple, interpretable
**Cons**: assumes spherical clusters; sensitive to k choice; sensitive to outliers

### Choosing k

#### Elbow Method
Plot WCSS vs k; pick k at "elbow" (diminishing returns).

```python
inertias = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k)
    inertias.append(kmeans.inertia_)
plt.plot(inertias)
# Visual inspection for elbow
```

**Issue**: subjective; elbow may be ambiguous

#### Silhouette Score
Measure how similar point is to own cluster vs others.

```
silhouette(i) = (b_i - a_i) / max(a_i, b_i)
```

where:
- a_i = mean distance to points in same cluster
- b_i = mean distance to points in nearest other cluster

Range: [-1, 1]; higher = better

```python
from sklearn.metrics import silhouette_score

best_k = max(range(2, 10), 
             key=lambda k: silhouette_score(X, KMeans(k).fit_predict(X)))
```

#### Gap Statistic
Compare WCSS to random uniform data; choose k with largest gap.

---

## Hierarchical Clustering

**Build tree** (dendrogram) of nested clusters; cut at desired level.

### Linkage Criteria

How to measure distance between clusters:

- **Single linkage**: min distance between any two points (prone to chaining)
- **Complete linkage**: max distance between any two points (compact clusters)
- **Average linkage**: mean distance (balanced; most commonly used)
- **Ward**: minimize variance increase when merging

```python
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

Z = linkage(X, method='ward')  # hierarchical clustering
dendrogram(Z)
plt.show()

# Cut dendrogram at height to get k clusters
from scipy.cluster.hierarchy import fcluster
labels = fcluster(Z, k, criterion='maxclust')
```

**Pros**: no k needed; interpretable dendrogram; flexible distance metrics
**Cons**: slow O(n²) or O(n³); can't undo merges; many parameters

---

## DBSCAN (Density-Based Spatial Clustering)

**Idea**: clusters = dense regions separated by sparse regions.

**Parameters**:
- eps: radius of neighborhood
- min_samples: minimum points to form dense region

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)
# -1 = noise/outlier; 0, 1, 2, ... = cluster labels
```

**Algorithm**:
1. For each point, find eps-neighborhood
2. If ≥ min_samples points → core point; start cluster
3. Expand cluster: include neighbors of core points (recursive)
4. Points not in any cluster = noise

**Pros**: finds arbitrary shapes; handles noise; no k needed
**Cons**: eps, min_samples tuning; poor on varying density

### HDBSCAN
Hierarchical DBSCAN; automatically chooses eps via minimum spanning tree.

```python
from sklearn.cluster import HDBSCAN

hdbscan = HDBSCAN(min_cluster_size=5)
labels = hdbscan.fit_predict(X)
```

---

## Gaussian Mixture Models (GMM)

**Probabilistic**: each cluster = Gaussian distribution; soft membership.

```
P(x) = Σ_k π_k · N(x | μ_k, Σ_k)
```

where π_k = mixture weight, N = normal distribution

**EM Algorithm**:
1. Initialize π, μ, Σ
2. E-step: compute responsibility (posterior probability of cluster for each point)
3. M-step: update π, μ, Σ using responsibilities (weighted MLE)

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=k, covariance_type='full')
labels = gmm.fit_predict(X)
responsibilities = gmm.predict_proba(X)  # soft membership [0,1]

# Choose k by BIC or AIC
bic_scores = [GaussianMixture(n_components=i).fit(X).bic(X) for i in range(1, 10)]
best_k = np.argmin(bic_scores) + 1
```

**Pros**: probabilistic; soft clustering; BIC/AIC for model selection
**Cons**: assumes Gaussian; slow EM iterations; can get stuck

---

## Other Methods

### K-Medoids (PAM)
Like K-means but centroid = actual data point (more robust to outliers).

```python
from sklearn_extra.cluster import KMedoids

kmedoids = KMedoids(n_clusters=k, metric='euclidean')
labels = kmedoids.fit_predict(X)
```

### Spectral Clustering
Use eigenvalues of affinity matrix; finds non-convex clusters.

```python
from sklearn.cluster import SpectralClustering

spec = SpectralClustering(n_clusters=k, affinity='nearest_neighbors')
labels = spec.fit_predict(X)
```

### OPTICS
DBSCAN variant; produces ordering of points; handles varying density.

---

## Dimensionality Reduction

### PCA
Project to principal components (maximize variance).

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels)
```

### t-SNE
Preserve local structure; for 2D visualization.

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30)
X_2d = tsne.fit_transform(X)
```

**Warning**: t-SNE only for visualization; don't use reduced data for clustering

### UMAP
Fast alternative to t-SNE; preserves both local + global structure.

```python
import umap

reducer = umap.UMAP(n_components=2)
X_2d = reducer.fit_transform(X)
```

---

## Evaluation Metrics

### Internal Metrics (no labels)

#### Silhouette Coefficient
Range: [-1, 1]; higher = better

#### Davies-Bouldin Index
Ratio of within-to-between cluster distances; lower = better

```python
from sklearn.metrics import davies_bouldin_score

db_index = davies_bouldin_score(X, labels)
```

#### Calinski-Harabasz Index
Ratio of between-cluster to within-cluster spread; higher = better

### External Metrics (with ground truth)

#### Purity
Fraction of correctly clustered points (assuming labels known)

#### Normalized Mutual Information (NMI)
Mutual information between predicted and true labels (normalized to [0,1])

```python
from sklearn.metrics import normalized_mutual_info_score

nmi = normalized_mutual_info_score(y_true, y_pred)
```

#### Adjusted Rand Index (ARI)
Similarity between partitions; adjusted for chance; range [-1, 1]

```python
from sklearn.metrics import adjusted_rand_score

ari = adjusted_rand_score(y_true, y_pred)
```

---

## Practical Workflow

1. **Preprocess**: normalize/scale, remove outliers
2. **Explore**: try multiple k values
3. **Choose algorithm**: 
   - Convex clusters, known k → K-means
   - Unknown k, arbitrary shapes → DBSCAN / HDBSCAN
   - Probabilistic model needed → GMM
   - High-dimensional → Spectral clustering
4. **Evaluate**: silhouette, Davies-Bouldin, visual inspection
5. **Interpret**: analyze cluster characteristics (mean features, size, etc.)

---

## Interview Key Points

- **K-means vs DBSCAN?** K-means: assumes spherical clusters, requires k, fast. DBSCAN: arbitrary shapes, no k, finds noise, slower.
- **Silhouette score interpretation?** Value [0,1]: higher = better. Negative = point closer to other cluster; check if k is appropriate.
- **How to choose k?** Elbow method (visual), Silhouette score (quantitative), domain knowledge, downstream task performance.
- **Hierarchical clustering: advantage?** Don't need to choose k beforehand; dendrograms interpretable; can cut at different levels.
- **GMM vs K-means?** GMM: soft membership (probabilistic); K-means: hard membership. GMM: model selection via BIC/AIC; K-means: direct optimization.
- **Why scale features before clustering?** Different scales → large-scale features dominate distance; normalize to equal weight.
- **DBSCAN: how to choose eps?** K-distance graph (plot sorted k-nearest distances); elbow = good eps value.
- **Evaluation without labels?** Silhouette, Davies-Bouldin, Calinski-Harabasz; visual inspection of reduced data; domain expert review.
- **Soft vs hard clustering?** Hard: each point assigned to one cluster (K-means, DBSCAN). Soft: membership probability (GMM, spectral).
