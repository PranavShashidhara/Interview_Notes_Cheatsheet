# Anomaly Detection

## Overview

**Anomaly**: observation deviating from normal pattern. Also: outlier, novelty, one-class problem.

**Challenge**: normal data far exceeds anomalies (imbalanced); anomalies often unknown.

**Applications**: fraud detection, network intrusion, equipment failure, medical diagnosis.

---

## Statistical Methods

### Z-Score
Flag points > 3 standard deviations from mean.

```python
mean = data.mean()
std = data.std()
z_scores = (data - mean) / std
anomalies = data[abs(z_scores) > 3]
```

**Pros**: simple, interpretable, fast
**Cons**: assumes normal distribution; univariate only

### Modified Z-Score (MAD)
Use median and MAD (median absolute deviation) instead (robust to outliers).

```python
median = data.median()
mad = np.median(np.abs(data - median))
mod_z = 0.6745 * (data - median) / mad
anomalies = data[abs(mod_z) > 3.5]
```

### Isolation Forest

**Idea**: anomalies isolated in feature space; use random partitioning.

```python
from sklearn.ensemble import IsolationForest

model = IsolationForest(contamination=0.1)  # expect 10% anomalies
predictions = model.fit_predict(X)  # -1 for anomalies, 1 for normal
```

**How**: random split features/thresholds; anomalies isolated quickly (few splits needed)
**Pros**: fast, no distance computation, handles high-dimensions
**Cons**: assumes anomalies isolated; poor on dense data

### Local Outlier Factor (LOF)

**Idea**: anomalies have lower density than neighbors.

```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20)
predictions = lof.fit_predict(X)
scores = -lof.negative_outlier_factor_  # higher = more anomalous
```

**Pros**: detects local anomalies; no global threshold
**Cons**: computationally expensive; requires neighbor search

### One-Class SVM

Learn boundary around normal data; points outside = anomalies.

```python
from sklearn.svm import OneClassSVM

svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)  # nu = expected anomaly fraction
predictions = svm.fit_predict(X)  # -1 for anomalies
```

**Pros**: max-margin boundary; high-dimensional friendly
**Cons**: slow; hyperparameter tuning needed

---

## Reconstruction-Based Methods

### Autoencoder

Encode input to low-dim → decode back. Anomalies reconstruct poorly.

```python
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Train on normal data only
# Anomaly score = reconstruction error
model = Autoencoder(input_dim, 8)
for x in normal_data:
    reconstructed = model(x)
    loss = MSE(reconstructed, x)
    loss.backward()

# Inference
reconstruction_errors = MSE(model(test_data), test_data)
anomalies = reconstruction_errors > threshold
```

**Pros**: learns nonlinear patterns; flexible architecture
**Cons**: prone to overfit if threshold not tuned; needs normal data for training

### Variational Autoencoder (VAE)

Probabilistic autoencoder; learns distribution of normal data.

```
loss = reconstruction_loss + KL_divergence(q(z|x) || p(z))
```

**Anomaly score**: -log p(x) (likelihood of data under learned model)

High reconstruction error + high KL → anomaly

### PCA

Project to principal components; anomalies have high reconstruction error.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=k)
X_reduced = pca.fit_transform(X)
X_reconstructed = pca.inverse_transform(X_reduced)

reconstruction_error = ((X - X_reconstructed) ** 2).sum(axis=1)
anomalies = reconstruction_error > threshold
```

**Pros**: simple, fast
**Cons**: linear; assumes anomalies in low-variance directions

---

## Time Series Anomalies

### Seasonal Decomposition
Decompose: Y(t) = Trend + Seasonal + Residual

Anomalies in residual (deviation from trend + seasonality).

```python
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(series, model='additive', period=12)
residuals = result.resid
anomalies = abs(residuals) > 3 * residuals.std()
```

**Pros**: handles seasonal patterns; interpretable
**Cons**: requires periodic data; parameter tuning

### ARIMA Residuals
Fit ARIMA; large residuals = anomalies.

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(series, order=(1,1,1))
result = model.fit()
residuals = result.resid
anomalies = abs(residuals) > 3 * residuals.std()
```

### Autoencoder on Sequences
Encode sequence of past k values; large reconstruction error = anomaly.

```python
# Input: (batch, seq_len, features)
# Autoencoder learns normal temporal patterns
reconstruction_errors = MSE(model(X), X)
```

---

## Evaluation Metrics

### Precision, Recall, F1
- **Precision**: % detected anomalies that are true
- **Recall**: % true anomalies detected
- **F1**: harmonic mean (balance precision & recall)

```python
from sklearn.metrics import precision_score, recall_score, f1_score

if_model = IsolationForest()
pred = if_model.predict(X)

precision = precision_score(y_true, (pred == -1).astype(int))
recall = recall_score(y_true, (pred == -1).astype(int))
```

### ROC-AUC
Threshold-independent; plot TPR vs FPR.

```python
from sklearn.metrics import roc_auc_score

# Anomaly scores (higher = more anomalous)
scores = -isolation_forest.score_samples(X)
auc = roc_auc_score(y_true, scores)
```

**Why ROC-AUC?**: handles class imbalance; threshold-agnostic

---

## Practical Considerations

### Threshold Selection
Set threshold based on business cost (false positives vs false negatives).

```python
# Example: credit card fraud
# False positive (block legitimate): lose $1 transaction
# False negative (miss fraud): lose $100 transaction

# Choose threshold optimizing: FP_cost * FP_rate + FN_cost * FN_rate
```

### Imbalanced Data
Anomalies typically 0.1% - 1% of data.

- Don't use accuracy (misleading)
- Use precision/recall/F1/ROC-AUC
- Adjust threshold or class weights

### Concept Drift
Normal patterns change over time (e.g., user behavior).

**Solution**: retrain model periodically; use adaptive methods

### Multiple Anomaly Types
Different anomalies may need different detectors.

```python
# Ensemble approach
iso_forest = IsolationForest()
lof = LocalOutlierFactor()
svm = OneClassSVM()

# Anomaly if majority vote anomaly
predictions = np.array([
    iso_forest.predict(X) == -1,
    lof.predict(X) == -1,
    svm.predict(X) == -1
])
anomalies = predictions.sum(axis=0) >= 2  # majority vote
```

---

## Interview Key Points

- **Statistical vs ML-based anomaly detection?** Statistical: fast, interpretable, assumes distribution. ML: flexible, captures complex patterns, needs more data.
- **Isolation Forest: why effective?** Anomalies isolated; require few splits to isolate (shorter trees); no distance calculation needed.
- **Autoencoder vs Statistical?** Autoencoder: learns nonlinear patterns, complex behavior. Statistical: simple baseline, interpretable, fast.
- **How to set threshold?** Business cost analysis; ROC curve knee point; or domain expertise.
- **Why ROC-AUC for imbalanced data?** Accuracy misleading (high baseline = always predict normal). ROC-AUC threshold-independent; fair comparison.
- **Time series anomalies: how detect seasonal?** Decompose into trend/seasonal/residual; anomalies in residual.
- **When to retrain anomaly detector?** Concept drift detected (false negative rate increasing); regular periodic retraining; user feedback loop.
- **Evaluate unsupervised anomaly detection?** No labels; use domain experts to label small sample. Precision-recall on labeled subset; ROC-AUC if possible.
- **Handle multiple anomaly types?** Ensemble detectors; clustering within anomalies; separate detectors per type.
