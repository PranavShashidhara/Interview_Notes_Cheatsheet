# Feature Engineering and Selection

## Overview

Feature engineering: domain knowledge → new features that improve model performance.
Feature selection: choose subset of features (reduce dimensionality, interpretability, training time).

**Impact**: often 80% of ML project effort; can outweigh model choice.

---

## Categorical Encoding

### One-Hot Encoding
Convert categorical variable with k categories into k binary columns.

```python
from sklearn.preprocessing import OneHotEncoder
# Color: [red, blue, green] → [1,0,0], [0,1,0], [0,0,1]
encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X_cat)
```

**Pros**: interpretable; works with most models
**Cons**: high dimensionality for high-cardinality features (thousands of categories)
**When**: low cardinality (< 50 categories)

### Label Encoding
Map categories to integers [0, k-1].

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_encoded = le.fit_transform(X_cat)  # [red, blue, green] → [2, 0, 1]
```

**Pros**: low dimensionality
**Cons**: implies ordinality (model thinks 2 > 1 > 0); tree-based OK, linear models not
**When**: high-cardinality, tree-based models, or ordinal categories

### Target Encoding (Mean Encoding)
Replace category with mean target value for that category.

```python
# For binary target: mean(y | category)
target_mean = df.groupby('category')['target'].mean()
X_encoded = X['category'].map(target_mean)
```

**Pros**: captures target-category relationship; low dimensionality; effective for tree models
**Cons**: target leakage risk; overfitting on rare categories
**Mitigation**: K-fold encoding (compute mean on fold-out set), smoothing (Laplace smoothing)

### Embedding Encoding
Learn low-dimensional embedding for each category (like Word2Vec).

```python
# Neural network: embedding layer (cat_id) → k-dimensional vector
embedding = Embedding(num_categories, embedding_dim)(category_ids)
```

**Pros**: captures category relationships; efficient for high-cardinality
**Cons**: requires neural network; needs training data
**When**: high cardinality (millions), neural model

### Hashing (Feature Hashing)
Hash category to fixed number of buckets.

```python
# Hash(category) % num_buckets → bucket_id
# Set that bucket to 1 (sparse one-hot)
```

**Pros**: handles unknown categories; fixed dimensionality; streaming data
**Cons**: collisions; no interpretability
**When**: high-cardinality online learning

---

## Numerical Feature Scaling

### Standardization (Z-score Normalization)
x_scaled = (x - mean) / std

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Properties**: mean 0, std 1; unbounded range (-inf, +inf)
**When**: linear models, SVM, neural networks (sensitive to scale)
**Preserves**: outliers

### Normalization (Min-Max Scaling)
x_scaled = (x - min) / (max - min) → [0, 1]

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

**Properties**: bounded [0, 1]; preserves zero
**When**: bounded features; neural networks with specific activations
**Issue**: sensitive to outliers (outlier shifts min/max)

### Robust Scaling
x_scaled = (x - median) / IQR

Uses median and interquartile range (IQR = Q3 - Q1).

**Properties**: robust to outliers; similar scale to standardization
**When**: data with outliers; tree-based (scale-invariant anyway)

### Log Transformation
x_scaled = log(x + 1)

**Use**: right-skewed distributions, wide range of values (e.g., income, counts)
```python
X_log = np.log1p(X)  # log1p avoids log(0)
```

**Effect**: compresses large values; expands small values

### Power Transformation
- **Box-Cox**: optimal power transformation (requires x > 0)
- **Yeo-Johnson**: variant that handles negative values

```python
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='box-cox')
X_transformed = pt.fit_transform(X)
```

---

## Feature Interactions & Polynomials

### Polynomial Features
Add higher-order terms: x, x², xy, etc.

```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)  # [x1, x2] → [1, x1, x2, x1², x1*x2, x2²]
```

**Pros**: captures non-linear relationships
**Cons**: curse of dimensionality; overfitting risk
**When**: small feature set (< 10), enough data, domain knowledge suggests interaction

### Interaction Terms (Manual)
Domain-specific: price × quantity, age × income, etc.

```python
X['price_qty_interaction'] = X['price'] * X['quantity']
X['age_income'] = X['age'] * X['income']
```

**Pros**: interpretable; domain-driven; avoids explosion of features
**Cons**: requires domain knowledge
**When**: clear business logic

### Binning & Bucketing
Convert continuous variable to categorical (ordinal or nominal).

```python
age_binned = pd.cut(age, bins=[0, 18, 35, 50, 65, 100], 
                     labels=['child', 'young', 'mid', 'senior', 'elderly'])
```

**Pros**: captures non-linearity; creates categorical feature
**Cons**: loses information; arbitrary bin boundaries
**When**: highly non-linear relationship; rare values

---

## Feature Selection

### Filter Methods (Univariate)
Rank features by statistical test; select top-k independent of model.

#### Correlation-based
```python
# Pearson correlation with target
corr = df.corr()['target'].abs().sort_values(ascending=False)
selected = corr[corr > threshold].index
```

#### Chi-squared (Classification)
```python
from sklearn.feature_selection import chi2
# Measures independence between categorical feature and target
scores, pvals = chi2(X_cat, y)
```

#### Mutual Information
```python
from sklearn.feature_selection import mutual_info_classif
scores = mutual_info_classif(X, y)
```

**Pros**: fast; model-agnostic; no overfitting
**Cons**: ignores feature interactions; univariate only

### Wrapper Methods (Model-based)
Train model repeatedly; select features that improve performance.

#### Forward Selection
```python
# Start with empty set; iteratively add features that most improve CV score
selected = []
remaining = all_features
for _ in range(k):
    best_f = argmax(score(selected + [f]) for f in remaining)
    selected.append(best_f)
    remaining.remove(best_f)
```

**Pros**: captures interactions
**Cons**: computationally expensive; greedy (no backtracking)

#### Backward Elimination
```python
# Start with all features; iteratively remove features that least hurt performance
selected = all_features
for _ in range(len(all_features) - k):
    worst_f = argmin(drop_in_score(selected - [f]) for f in selected)
    selected.remove(worst_f)
```

#### Recursive Feature Elimination (RFE)
```python
from sklearn.feature_selection import RFE
rfe = RFE(estimator=model, n_features_to_select=k, step=1)
X_selected = rfe.fit_transform(X, y)
```

**Works**: iteratively removes features ranked least important by model (e.g., SVM weights)

### Embedded Methods
Feature selection as part of model training.

#### Tree Importance
```python
model = RandomForestClassifier()
model.fit(X, y)
importances = model.feature_importances_
```

**How**: trees rank features by reduction in impurity (Gini, entropy)
**Pros**: captures interactions; fast
**Cons**: biased toward high-cardinality; doesn't distinguish correlated features

#### Regularization (L1/L2)
```python
# L1 regularization (Lasso) forces irrelevant weights to zero
model = LogisticRegression(penalty='l1', solver='liblinear')
model.fit(X_scaled, y)
selected = X.columns[model.coef_[0] != 0]
```

**Pros**: interpretable; feature elimination during training
**Cons**: correlated features → arbitrary selection

#### Permutation Importance
```python
from sklearn.inspection import permutation_importance
result = permutation_importance(model, X_val, y_val)
# Drop in score when feature values shuffled
```

**Pros**: model-agnostic; captures interactions
**Cons**: expensive (requires retraining)

---

## Dimensionality Reduction

### PCA (Principal Component Analysis)
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=k)
X_reduced = pca.fit_transform(X_scaled)
```

**How**: find k orthogonal directions (principal components) of maximum variance
**Interpretability**: new features are linear combinations (hard to interpret)
**When**: many correlated features; linear relationships

### t-SNE / UMAP (Visualization)
```python
from sklearn.manifold import TSNE
X_2d = TSNE(n_components=2).fit_transform(X)
```

**Use**: 2D visualization for exploratory analysis (not for training models)

---

## Domain-Specific Strategies

### Time-Based Features (Temporal)
```python
X['hour'] = df['timestamp'].dt.hour
X['day_of_week'] = df['timestamp'].dt.dayofweek
X['is_weekend'] = df['timestamp'].dt.dayofweek >= 5
X['days_since_event'] = (now - df['date']).dt.days
```

### Text Features (NLP)
```python
# TF-IDF, word counts, length, sentiment
X['text_length'] = df['text'].str.len()
X['word_count'] = df['text'].str.split().str.len()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=100)
X_text = tfidf.fit_transform(df['text'])
```

### Geographic Features (Location)
```python
X['lat'], X['lon'] = geocode(df['address'])
X['distance_to_city'] = haversine(lat, lon, city_lat, city_lon)
```

---

## Common Mistakes & Best Practices

### Data Leakage
**Problem**: target information leaks into training features.

```python
# WRONG: scale using full dataset (train + test)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # leakage!

# RIGHT: fit scaler on train only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # same scaler
```

### Target Leakage in Encoding
```python
# WRONG: compute target encoding on full dataset
target_mean = df.groupby('category')['target'].mean()
X['encoded'] = X['category'].map(target_mean)  # leakage!

# RIGHT: K-fold encoding
for fold in k_folds:
    target_mean = df[train_fold].groupby('category')['target'].mean()
    X.loc[val_fold, 'encoded'] = X.loc[val_fold, 'category'].map(target_mean)
```

### Over-engineering
**Problem**: too many features → overfitting → poor generalization.

**Solution**: start simple; add features incrementally; monitor CV score.

### Ignoring Class Imbalance in Encoding
```python
# For rare categories: smoothing
target_mean = (df.groupby('category')['target'].sum() + alpha) / \
              (df.groupby('category').size() + 2*alpha)
```

---

## Practical Workflow

1. **EDA**: visualize distributions, missing patterns, outliers
2. **Handle missing**: impute or drop (domain-specific)
3. **Scale numericals**: StandardScaler (linear models), MinMaxScaler (neural nets)
4. **Encode categoricals**: one-hot (low cardinality), target (high cardinality)
5. **Create interactions**: domain knowledge first; polynomial only if justified
6. **Select features**: start with filter (fast); refine with wrapper/embedded
7. **Monitor**: track CV score; avoid overfitting
8. **Iterate**: model + features co-evolve

---

## Interview Key Points

- **Why scale features?** Linear models assume equal scale; gradient descent converges faster; distance-based algorithms (KNN, SVM) sensitive to scale.
- **One-hot vs target encoding?** One-hot: low cardinality, no assumptions. Target: high cardinality, captures target-feature relationship; risk of leakage.
- **How to handle high-cardinality categoricals?** Target encoding, hashing, embeddings, grouping rare categories into "Other".
- **When to use polynomial features?** Small feature set + suspected non-linear relationships; beware curse of dimensionality.
- **Filter vs wrapper selection?** Filter: fast, model-agnostic; ignores interactions. Wrapper: captures interactions; slow. Use filter first, then wrapper.
- **Why is data leakage problematic?** Model looks good in dev but fails in production; violates train/test independence.
- **PCA: when to use?** Many correlated features; reduce dimensionality. Cost: new features hard to interpret.
- **Permutation vs tree importance?** Tree: built-in, fast, biased. Permutation: model-agnostic, captures interactions, expensive.
- **How to avoid overfitting in feature engineering?** Use CV; don't fit encoders/scalers on full data; add features incrementally; regularization (L1/L2).
