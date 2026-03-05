# Metrics: Classification, Regression, Ranking, Generation, and Model Evaluation

## Confusion Matrix (Binary Classification)

```
                  Predicted Positive    Predicted Negative
Actual Positive        TP                    FN
Actual Negative        FP                    TN
```

- **TP**: True Positive — correctly predicted positive
- **FP**: False Positive — predicted positive, actually negative (Type I error)
- **FN**: False Negative — predicted negative, actually positive (Type II error)
- **TN**: True Negative — correctly predicted negative

---

## Classification Metrics

### Accuracy
Accuracy = (TP + TN) / (TP + FP + FN + TN)

When to avoid: imbalanced classes. A model predicting all majority class can have high accuracy with zero utility.

### Precision
Precision = TP / (TP + FP)

"Of everything predicted positive, how many actually are?" Important when FP is costly (e.g., spam filter).

### Recall (Sensitivity / True Positive Rate)
Recall = TP / (TP + FN)

"Of all actual positives, how many did we catch?" Important when FN is costly (e.g., cancer screening).

### F1 Score
F1 = 2 * Precision * Recall / (Precision + Recall) = 2*TP / (2*TP + FP + FN)

Harmonic mean of precision and recall. Use when you need a single metric balancing both.

### F-beta Score
F_beta = (1 + beta^2) * Precision * Recall / (beta^2 * Precision + Recall)

- beta > 1: weights recall more (miss fewer positives)
- beta < 1: weights precision more (fewer false alarms)

### Specificity (True Negative Rate)
Specificity = TN / (TN + FP)

### False Positive Rate
FPR = FP / (FP + TN) = 1 - Specificity

### ROC Curve
Plots TPR (Recall) vs FPR at all decision thresholds.

**AUC-ROC**: Area Under ROC curve
- 0.5 = random classifier
- 1.0 = perfect classifier
- Interpretation: probability that a randomly chosen positive ranks higher than a randomly chosen negative

**When to prefer ROC-AUC**: balanced classes, care about ranking, not a specific threshold.

### Precision-Recall Curve and PR-AUC
Plots Precision vs Recall at all thresholds. More informative than ROC when classes are severely imbalanced (FP pool is small, making FPR low by default).

**Average Precision (AP)**: weighted mean of precisions at each threshold. mAP extends this to multiple classes/queries.

### Macro vs Micro vs Weighted Averaging
- **Macro**: compute metric per class, take unweighted mean — treats all classes equally
- **Micro**: aggregate TP, FP, FN across all classes first, then compute — dominated by frequent classes
- **Weighted**: average weighted by class support — accounts for imbalance

Used in your Multilingual Toxicity project: per-language F1 + ROC-AUC per intent class.

### Log Loss (Cross-Entropy Loss)
Log Loss = -1/n * sum(y_i * log(p_i) + (1-y_i) * log(1-p_i))

Penalizes confident wrong predictions heavily. Lower = better. Used directly as training loss in classification.

### Matthews Correlation Coefficient (MCC)
MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

Range [-1, 1]. 1 = perfect, 0 = random, -1 = inverse. Best single metric for imbalanced binary classification.

---

## Multiclass Classification

### Confusion Matrix Extension
k x k matrix for k classes.

### Per-Class Metrics
Compute precision, recall, F1 for each class treating it as binary (one-vs-rest).

### Cohen's Kappa
Kappa = (p_o - p_e) / (1 - p_e)

Compares observed accuracy vs expected accuracy by chance. More robust than accuracy for imbalanced multiclass.

---

## Regression Metrics

### Mean Absolute Error (MAE)
MAE = 1/n * sum |y_i - y_hat_i|

Interpretation: average absolute error in target units. Robust to outliers. All errors weighted equally.

### Mean Squared Error (MSE)
MSE = 1/n * sum (y_i - y_hat_i)^2

Penalizes large errors more. Differentiable everywhere. Used in your Netflix project (Train MSE 0.5854, Test MSE 0.842).

### Root Mean Squared Error (RMSE)
RMSE = sqrt(MSE)

In target units. Comparable to MAE but more sensitive to outliers. Netflix Prize benchmark was RMSE ~0.910.

### R-squared (Coefficient of Determination)
R^2 = 1 - SS_res / SS_tot = 1 - sum(y_i - y_hat_i)^2 / sum(y_i - y_bar)^2

0 = predicting mean, 1 = perfect. Can be negative if model is worse than mean predictor.

### Mean Absolute Percentage Error (MAPE)
MAPE = 100/n * sum |y_i - y_hat_i| / |y_i|

Undefined when y_i = 0. Scale-independent, useful for comparing across datasets.

---

## Ranking and Recommendation Metrics

### Precision@k
Fraction of top-k recommended items that are relevant.
P@k = |relevant items in top-k| / k

### Recall@k
Fraction of all relevant items that appear in top-k.
R@k = |relevant items in top-k| / |total relevant items|

### Mean Average Precision (MAP)
Average of AP scores across all queries.
AP = sum over k [P@k * rel(k)] / |relevant items|

### Normalized Discounted Cumulative Gain (NDCG)
DCG@k = sum from i=1 to k of rel_i / log2(i+1)
NDCG@k = DCG@k / IDCG@k (ideal DCG)

Accounts for position — relevant items ranked higher contribute more. Used in recommender evaluation.

### Hit Rate
Fraction of users for whom at least one relevant item appears in top-k.

---

## Image Generation / CV Metrics (Your Brain MRI Project)

### FID (Frechet Inception Distance)
Measures distance between feature distributions of real and generated images using Inception network.
FID = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2*(Sigma_r * Sigma_g)^0.5)

Lower FID = more realistic images. Your results: FID 219 (baseline) → 58 (M3 with MAT).

### KID (Kernel Inception Distance)
Uses maximum mean discrepancy (MMD) between Inception features. Unlike FID, unbiased estimator with confidence intervals. Lower = better. Your results: KID 0.791 → 0.133.

### SSIM (Structural Similarity Index)
Measures similarity between two images across luminance, contrast, and structure.
Range [0,1], higher = more similar. Your results: SSIM 0.26 → 0.72.

### Dice Score (Segmentation)
Dice = 2|A ∩ B| / (|A| + |B|) = 2*TP / (2*TP + FP + FN)

Measures overlap between predicted and ground truth segmentation mask. Range [0,1]. Your results: GenDice 0.154 → 0.688.

### DiffMapIoU
Intersection over Union of difference maps — measures edit locality in counterfactual generation. Your results: 0.023 → 0.157.

### TumorResidual
Measures how much tumor remains in counterfactual (tumor-removed) images. Lower = better. Your results: 1.625 → 0.546 via MAT.

---

## NLP Generation Metrics

### BLEU (Bilingual Evaluation Understudy)
Precision of n-grams in generated text vs references, with brevity penalty.
BLEU = BP * exp(sum_n w_n * log p_n)

Range [0,1]. Good for translation. Insensitive to meaning — purely lexical.

### ROUGE
- **ROUGE-N**: n-gram recall overlap
- **ROUGE-L**: longest common subsequence
- **ROUGE-S**: skip-bigram overlap

Used for summarization evaluation.

### METEOR
Aligns generated and reference at word level, considers synonyms and stemming. Correlates better with human judgments than BLEU.

### BERTScore
Computes cosine similarity between BERT contextual embeddings of generated and reference tokens. Captures semantic similarity beyond exact overlap.

### Perplexity
PP = exp(H(p, q)) = exp(-1/N * sum log p(w_i))

Measures how well a language model predicts held-out text. Lower = better. Intrinsic LLM metric.

---

## Clustering Metrics

### Silhouette Score
s(i) = (b(i) - a(i)) / max(a(i), b(i))
a(i) = average intra-cluster distance, b(i) = average distance to nearest other cluster.
Range [-1, 1]. Higher = better defined clusters.

### Davies-Bouldin Index
Ratio of within-cluster scatter to between-cluster separation. Lower = better.

### Inertia (WCSS)
Sum of squared distances of samples to their cluster center. Used for elbow method.

### Adjusted Rand Index (ARI)
Measures similarity between cluster assignments and ground truth labels, adjusted for chance. Range [-1, 1].

---

## LLM and Embedding Metrics (Your RAG and LLM Projects)

### Faithfulness
Does the generated answer contain only information supported by the retrieved context? Measured via entailment or LLM-as-judge.

### Answer Relevance
Is the generated answer relevant to the question? Measured via cosine similarity of question embedding to answer embedding.

### Context Precision
Fraction of retrieved chunks that are relevant to the query. Higher = cleaner retrieval.

### Context Recall
Fraction of relevant information from ground truth that was present in retrieved context.

### RAGAS Framework
Combines faithfulness, answer relevance, context precision, context recall into a unified RAG evaluation.

### Embedding Quality Metrics
- **Cosine Similarity**: dot product of unit vectors; measures directional similarity; range [-1,1]
- **Euclidean Distance**: straight-line distance in embedding space
- **MRR (Mean Reciprocal Rank)**: 1/rank of first relevant retrieved document, averaged over queries

---

## Model Evaluation Summary Table

| Task | Primary Metric | When to Use Alternatives |
|---|---|---|
| Binary Classification (balanced) | ROC-AUC, F1 | MCC for severe imbalance |
| Binary Classification (imbalanced) | PR-AUC, F1, MCC | Avoid accuracy |
| Multiclass | Macro F1, weighted F1 | Cohen's Kappa |
| Regression | RMSE, MAE, R^2 | MAPE for scale-independence |
| Ranking | NDCG, MAP, MRR | P@k, R@k |
| Generation (NLP) | BLEU, ROUGE, BERTScore | Perplexity (intrinsic) |
| Clustering | Silhouette, ARI | Inertia (elbow only) |
| Image Generation | FID, KID, SSIM | Task-specific (Dice, IoU) |
| RAG | RAGAS (faithfulness, context recall) | Custom LLM-as-judge |
