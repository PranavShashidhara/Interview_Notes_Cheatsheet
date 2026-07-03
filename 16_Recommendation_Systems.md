# Recommendation Systems

## Overview

Recommendation systems predict user preferences for items. Core task: given user-item interactions, predict ratings/rankings for unseen items.

**Business Value**: Netflix, Amazon, Spotify, Meta rely on recommendations for engagement and revenue.

---

## Problem Formulation

### Matrix Factorization Perspective
User-item interaction matrix R (m users × n items). Most entries missing.
Goal: predict missing entries; rank top-k items for each user.

### Key Metrics
- **Precision@k**: fraction of top-k items user likes
- **Recall@k**: fraction of liked items in top-k recommendations
- **NDCG@k**: penalizes bad ranking of relevant items
- **Hit Rate**: any recommendation liked user
- **Diversity**: recommend different items across users

---

## Collaborative Filtering (CF)

### User-User Similarity
Find similar users; recommend items they liked.

```python
# Cosine similarity between user rating vectors
sim(u1, u2) = cos(r_u1, r_u2)
# Predict: rating of u1 for item i = weighted avg of similar users' ratings
r_u1_i = mean(r_u2_i for similar users u2)
```

**Pros**: simple, interpretable
**Cons**: sparse data, cold-start problem

### Item-Item Similarity
Find similar items; if user liked item A, recommend similar items.

```python
sim(i1, i2) = cos(r_i1, r_i2)  # similarity in rating patterns
```

**Pros**: more stable than user-user (item preferences change slower)
**Cons**: still sparse, limited scope

### Matrix Factorization (SVD, NMF)

Decompose R ≈ U × V^T where:
- U: m × k (user latent factors)
- V: n × k (item latent factors)
- k: latent dimension (typically 10-100)

**SVD (Singular Value Decomposition)**:
R = U × Σ × V^T
- Exact but doesn't handle missing values
- Impute missing entries first, then SVD

**Regularized Matrix Factorization**:
```
min ||R - U×V^T||^2 + lambda*(||U||^2 + ||V||^2)
```
- Alternating Least Squares (ALS): efficient, scalable
- Stochastic Gradient Descent (SGD): memory-efficient

**Bias terms**:
```
r_ui = mu + b_u + b_i + <U_u, V_i>
```
where mu = global mean, b_u = user bias, b_i = item bias

---

## Content-Based Filtering

Recommend items similar to user's past preferences.

```python
# Item features: genre, keywords, price, etc.
sim(item_i, item_j) = cosine(features_i, features_j)
# Recommend: rank items by similarity to user's liked items
score_u_i = sum(sim(i, past_liked_j) for j in user's history)
```

**Pros**: no cold-start for items; explainable (features drive recommendations)
**Cons**: limited novelty; requires good features

**Hybrid**: combine user features (age, location) + item features

---

## Deep Learning Approaches

### Neural Collaborative Filtering (NCF)

Replace dot product with neural network:
```python
user_emb = embed(user_id)  # learned embedding
item_emb = embed(item_id)
concat = [user_emb, item_emb]
hidden = ReLU(Dense(concat))  # multiple layers
output = sigmoid(Dense(hidden))  # binary: like/dislike
```

**Advantages**: captures non-linear interactions; flexible architecture

### Two-Tower Model (Dual Encoder)

```
User Tower                Item Tower
  user_id                   item_id
    |                           |
  embed                       embed
    |                           |
 MLP(user_features)    MLP(item_features)
    |                           |
  user_emb (d-dim)        item_emb (d-dim)
    \                         /
     \       dot product     /
      \------> score
```

**Inference**: offline compute item embeddings; online nearest neighbor search (FAISS, Pinecone)
**Fast**: 100ms-level latency for ranking

### Sequence Models (RNN/Transformer)

Capture temporal dynamics: user's next item depends on past sequence.

```python
seq = [item_t-2, item_t-1, item_t]
embeddings = [embed(i) for i in seq]
hidden = LSTM(embeddings)  # or Transformer
next_item_logits = matmul(hidden, item_embeddings.T)
```

**Applications**: next-song prediction (Spotify), next-video (YouTube)

---

## Ranking & Re-ranking

### Learning-to-Rank (LTR)

1. **Retrieval stage**: retrieve top-100 candidates (fast, recall-focused)
2. **Ranking stage**: score candidates with expensive model; return top-10

```python
# Retrieval: fast collaborative filtering
candidates = retriever(user_id, k=100)

# Ranking: expensive neural model
scores = ranking_model(user_features, [item_features for item in candidates])
ranked = sort_by_score(candidates, scores)
```

### Diversity & Freshness

**Diversity penalty**: don't recommend similar items consecutively
```python
score_final = score_ml + alpha*diversity_penalty - beta*time_since_new
```

**Exploration**: recommend new/diverse items with small probability (ε-greedy)

### Cold-Start & Context

**New users**: recommend popular items, ask for preferences
**New items**: content-based initial score, then collaborative signal
**Context**: time-of-day, device, location modify scores

```python
score = base_score + context_boost(user_context, item, time)
```

---

## Implicit Feedback & Ranking

Many systems only have implicit signals: clicks, watches, purchases (no ratings).

**Binary signals**: user-item pair = (1 liked, 0 unknown)
- Unknown ≠ disliked (many untouched items)
- **Weighted loss**: penalize negative predictions less; use samplers

**Ranking loss** (vs rating loss):
```python
# BPR (Bayesian Personalized Ranking)
loss = -log(sigmoid(score_pos - score_neg))
# Penalize: liked items ranked lower than unliked
```

---

## Evaluation & A/B Testing

### Offline Metrics
- **Precision@k, Recall@k, NDCG@k**: on held-out test set
- **MAP (Mean Average Precision)**: average precision across cutoffs

**Issue**: offline metrics don't capture novelty, serendipity, or business metrics

### Online Metrics (A/B Testing)
- **CTR (Click-Through Rate)**: did user click the recommendation
- **Conversion Rate**: did user purchase
- **DAU (Daily Active Users)**: engagement
- **Revenue per User**: business metric

**Key challenge**: short-term (clicks) vs long-term value (engagement, diversity)

### Pitfalls
- **Popularity bias**: models learn to recommend popular items (safe but boring)
- **Filter bubble**: reinforce existing preferences; miss discovery
- **Position bias**: clicks biased toward top positions (offline ≠ online)

---

## Your Projects / Resume Points

### E-commerce Recommendation
- Implicit feedback (clicks, purchases)
- Product features: category, price, ratings
- User context: browsing history, location, device
- Real-time serving: candidate generation + ranking

### Music/Video Recommendation
- Temporal dynamics (sequence models)
- Cold-start handling
- Diversity: don't recommend same artist/genre repeatedly
- Freshness: promote new content

---

## Interview Key Points

- **CF vs content-based?** CF: learns user-item interactions; better for novelty. Content: needs features; good for cold-start items. Hybrid usually best.
- **Why matrix factorization?** Captures latent factors; scalable; low-rank assumption reasonable for most data.
- **How to handle sparse data?** Regularization (L2), factorization (projects to k-dim), sampling negative examples carefully.
- **What is cold-start?** New user/item has no interactions. Solutions: content features, popularity baseline, ask preferences, exploration.
- **Implicit vs explicit feedback?** Implicit: click ≠ dislike (only know likes). Explicit: ratings. Implicit needs ranking loss (BPR); larger scale.
- **Two-tower model advantage?** Separates user and item representations; enables efficient serving (precompute item embeddings, FAISS search).
- **How to avoid popularity bias?** Diversity penalty during ranking, exploration ε-greedy, calibration (recommend unpopular items with controlled rate).
- **Why NDCG over Precision/Recall?** Precision/Recall treat all positions equally. NDCG: top positions matter more (logarithmic discount).
- **Online vs offline metrics?** Offline: can optimize easily. Online: ground truth (clicks, revenue) but noisy, requires A/B test, longer to iterate.
