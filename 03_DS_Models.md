# Data Science Models: Internal Workings & Interview Guide

## Linear Regression

### How It Works
Fits a hyperplane y = theta_0 + theta_1*x_1 + ... + theta_n*x_n by minimizing Mean Squared Error (MSE).

**Closed-form (OLS)**: theta = (X^T X)^(-1) X^T y
**Gradient Descent**: theta := theta - alpha * (1/m) * X^T(X*theta - y)

### Assumptions
Linearity, independence, homoscedasticity, normality of residuals, no multicollinearity.

### Regularized Variants
- **Ridge**: adds L2 penalty; no closed-form sparsity; handles collinearity
- **Lasso**: adds L1 penalty; produces sparse solutions; feature selection built-in
- **Elastic Net**: L1 + L2; handles correlated features; used in your MediAssist/toxicity work implicitly

---

## Logistic Regression

### How It Works
Models P(y=1|x) = sigmoid(theta^T x) = 1 / (1 + exp(-theta^T x))

**Loss**: Binary Cross-Entropy: L = -[y log(p) + (1-y) log(1-p)]
**No closed form**: optimized via gradient descent or Newton-Raphson.

### Multiclass
- **One-vs-Rest**: train k binary classifiers
- **Softmax**: single model with softmax output

### Decision Boundary
Linear — logistic regression is a linear classifier despite the name.

---

## Decision Trees

### How It Works
Recursively splits data on feature thresholds that maximize information gain.

**Splitting Criteria**:
- **Gini Impurity**: 1 - sum(p_i^2) — used in CART
- **Information Gain (Entropy)**: H(parent) - weighted average H(children)
- **Variance Reduction** (regression)

**Stopping**: max depth, min samples per leaf, min impurity decrease.

### Advantages
Interpretable, handles nonlinear relationships, no feature scaling needed.

### Disadvantages
High variance (overfits easily), unstable — small data changes alter tree structure.

---

## Random Forest

### How It Works
Ensemble of decision trees using **Bagging** (Bootstrap Aggregating) + **Random Feature Subsets**.

1. Sample n data points with replacement (bootstrap sample)
2. At each split, consider only sqrt(p) or log2(p) random features
3. Train a tree fully (or until stopping criterion)
4. Aggregate: majority vote (classification), mean (regression)

### Key Properties
- Reduces variance without increasing bias (vs single tree)
- Feature importance: mean decrease in impurity across all trees
- Out-of-bag (OOB) error: free validation on non-bootstrapped samples (~37%)

---

## Gradient Boosting (XGBoost / LightGBM)

### How It Works
Sequential ensemble: each tree corrects residuals of the previous.

**Algorithm**:
1. Initialize: F_0(x) = argmin_gamma sum L(y_i, gamma)
2. For m = 1 to M:
   a. Compute pseudo-residuals: r_im = -[dL/dF(x_i)] at F = F_{m-1}
   b. Fit tree h_m to residuals
   c. F_m(x) = F_{m-1}(x) + eta * h_m(x)

**Loss**: any differentiable loss function (logistic, MSE, etc.)

### XGBoost Improvements
- Second-order Taylor expansion of loss
- L1 + L2 regularization on tree weights
- Approximate split finding
- Column subsampling

### LightGBM (Used in your Debt Default project)
- **GOSS** (Gradient-based One-Side Sampling): keeps large-gradient samples, randomly drops small-gradient samples
- **EFB** (Exclusive Feature Bundling): bundles mutually exclusive sparse features
- **Leaf-wise** growth (vs level-wise in XGBoost): deeper trees, lower loss, but can overfit
- ~10x faster than XGBoost on large datasets
- Best model in your Debt Default Prediction (99% accuracy, 97% macro F1)

---

## Support Vector Machines (SVM)

### How It Works
Finds the hyperplane that maximizes the margin between classes.

**Objective** (hard margin): minimize ||w||^2 subject to y_i(w^T x_i + b) >= 1

**Soft margin (C parameter)**: allows misclassification; C controls trade-off between margin and violations.

### Kernel Trick
Maps data to high-dimensional space implicitly via kernel function K(x_i, x_j):
- **Linear**: K(x,z) = x^T z
- **RBF (Gaussian)**: K(x,z) = exp(-gamma ||x-z||^2)
- **Polynomial**: K(x,z) = (x^T z + c)^d

### SVR (Support Vector Regression)
Fit within epsilon-tube; points outside tube incur loss.

---

## k-Nearest Neighbors (kNN)

### How It Works
Non-parametric, instance-based learning. Predicts by majority vote (classification) or mean (regression) of k nearest training points by Euclidean (or other) distance.

**No training** — all computation at inference. O(nd) per query.

### Key Considerations
- Feature scaling required (distance-sensitive)
- k selection: low k = high variance, high k = high bias
- Curse of dimensionality: distances become uniform in high dimensions

---

## Naive Bayes

### How It Works
Applies Bayes' theorem with strong (naive) independence assumption among features.

P(y|x_1,...,x_n) proportional to P(y) * product of P(x_i|y)

**Variants**:
- **Gaussian NB**: assumes P(x_i|y) is Gaussian
- **Multinomial NB**: for word counts (NLP)
- **Bernoulli NB**: for binary features

Fast, works well with small data and high-dimensional text.

---

## k-Means Clustering

### How It Works
1. Initialize k centroids randomly (or via k-means++)
2. Assign each point to nearest centroid
3. Recompute centroids as cluster means
4. Repeat until convergence

**Objective**: minimize within-cluster sum of squares (WCSS) = sum over k sum over x_i in C_k of ||x_i - mu_k||^2

**k-means++**: smart initialization — choose first centroid randomly, then each subsequent centroid proportional to squared distance from nearest existing centroid. Improves convergence and quality.

### Selecting k
- **Elbow method**: plot WCSS vs k; find elbow
- **Silhouette score**: (b - a) / max(a, b) where a = intra-cluster distance, b = nearest-cluster distance

---

## DBSCAN

### How It Works
Density-based clustering. Parameters: epsilon (neighborhood radius), min_samples.

- **Core point**: has >= min_samples neighbors within epsilon
- **Border point**: within epsilon of a core point but fewer neighbors
- **Noise**: not within epsilon of any core point

**Advantages**: finds arbitrarily shaped clusters, automatically identifies noise/outliers, no need to specify k.

---

## Principal Component Analysis (PCA)

### How It Works
1. Standardize data
2. Compute covariance matrix Sigma = (1/n) X^T X
3. Compute eigenvectors and eigenvalues of Sigma
4. Sort by eigenvalue (descending) — these are principal components
5. Project: Z = X * W (W = top k eigenvectors)

**Explained variance ratio**: proportion of variance captured by each PC.

**When to use**: dimensionality reduction, visualization, remove multicollinearity, noise reduction.

### SVD Connection
PCA is equivalent to SVD of centered X: X = U Sigma V^T. Principal components are columns of V.

---

## Matrix Factorization (Collaborative Filtering)

### How It Works (Your Netflix Project)
Decompose user-item rating matrix R (m x n) into:
R ≈ U * V^T where U is m x k (user latent factors), V is n x k (item latent factors)

**Objective**: minimize sum over observed (i,j) of (r_ij - u_i^T v_j)^2 + regularization

**Optimization**: Alternating Least Squares (ALS) or SGD.

**Results in your project**: Test RMSE 0.917, comparable to Netflix Prize winning benchmark of ~0.910.

### Restricted Boltzmann Machine (RBM)
Energy-based generative model with visible and hidden units. Trained via Contrastive Divergence. Used as collaborative filter in your Netflix project. Learns latent user/item representations.

---

## Bayesian Optimization (Optuna — Your Resume)

### How It Works
1. Sample initial points randomly
2. Fit surrogate model (Gaussian Process or TPE — Tree-structured Parzen Estimator)
3. Use acquisition function to select next evaluation point:
   - **Expected Improvement (EI)**: E[max(f(x) - f_best, 0)]
   - **Upper Confidence Bound (UCB)**: mu(x) + kappa * sigma(x)
4. Evaluate true objective at selected point
5. Update surrogate; repeat

**Advantage over grid/random search**: exploits known good regions while exploring uncertain regions. More efficient for expensive black-box functions.

Used in your Debt Default project: +20% minority class accuracy improvement.

---

## Model Selection and Validation

### Cross-Validation
- **k-fold**: split into k folds; train on k-1, test on 1; rotate; average metrics
- **Stratified k-fold**: preserves class distribution per fold (critical for imbalanced data)
- **Leave-One-Out (LOO)**: k = n; high variance, low bias estimate; expensive

### Bias-Variance Tradeoff
Total Error = Bias^2 + Variance + Irreducible Noise

- **High bias** (underfitting): model too simple; train and val error both high
- **High variance** (overfitting): model too complex; low train error, high val error
- **Solution**: regularization, more data, simpler model, ensembling

### Model Comparison
- Use paired t-test on cross-validated scores
- Report confidence intervals, not just point estimates
- Use Friedman test + Nemenyi post-hoc for comparing multiple models across datasets

---

## Key Interview Questions

**Why does Random Forest reduce variance but not bias?**
Each tree is high-variance, low-bias (deep). Averaging uncorrelated trees reduces variance. Correlation is reduced by random feature subsets. Bias is unchanged since each tree is as expressive.

**Why is LightGBM faster than XGBoost?**
GOSS reduces data per iteration; EFB reduces feature count; leaf-wise growth finds better splits per leaf count; histogram-based splitting avoids sorting.

**What is the difference between bagging and boosting?**
Bagging trains models in parallel on bootstrap samples; reduces variance. Boosting trains sequentially, each correcting predecessor's errors; reduces bias.

**When would you use SVM over tree methods?**
High-dimensional sparse data (text), when a clear margin exists, small-to-medium datasets. Tree methods generally win on tabular data at scale.

**What happens when k-means fails?**
Converges to local optima; sensitive to initialization and scale. Use k-means++, multiple restarts, and always standardize features.
