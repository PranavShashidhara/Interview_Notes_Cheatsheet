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

## Deep Learning Ranking Models

### DLRM (Deep Learning Recommendation Model — Meta)
- **Dense features** → MLP → dense embedding
- **Sparse features** (categorical) → embedding lookup
- **Feature interactions**: dot product between all embedding pairs
- Final MLP → prediction
- Standard architecture behind Meta's ads and recommendation systems

### DeepFM
- Combines **factorization machine** (2nd-order feature interactions) with **deep neural network**
- FM captures pairwise interactions; DNN captures higher-order
- No manual feature engineering needed

### DCN-v2 (Deep & Cross Network)
- **Cross network**: explicit feature crosses at each layer: `x_{l+1} = x_0 ⊙ (W · x_l + b) + x_l`
- **Deep network**: standard MLP for implicit interactions
- Efficient, bounded-degree feature interactions without exponential blowup

---

## Contextual Bandits for Exploration

### Why Bandits?
- Standard A/B tests are wasteful: you keep showing bad variants to 50% of users
- Bandits allocate more traffic to better variants while still exploring

### Algorithms
- **LinUCB**: linear model per arm, exploration via upper confidence bound on reward estimate
- **Thompson Sampling**: maintain posterior over reward for each arm, sample and pick best. Naturally balances explore/exploit.
- **Contextual**: features about the user/context inform the reward estimate per arm

### When to Use
- Content recommendation (homepage layout, article selection)
- Ad creative selection
- Model selection in ensemble serving
- When you have many variants and limited traffic

---

## Debiasing Logged Data

### Position Bias
- Items shown in top positions get more clicks regardless of relevance
- Naive models learn position → clicks, not quality → clicks

### Solutions
- **Inverse Propensity Scoring (IPS)**: weight each click by 1/P(position). Unbiased but high variance.
- **Position feature**: include position as a feature during training, set to 0 at inference
- **DLA (Dual Learning Algorithm)**: jointly learn relevance model and position bias model

---

## ML System Design: YouTube-Scale Recommendations

### Stage 1: Candidate Generation (~1000 candidates, <50ms)
- Two-tower model: user tower (history, demographics) + item tower (metadata, embeddings)
- ANN retrieval from item index (FAISS, ScaNN)
- Multiple candidate sources: similar items, trending, personalized

### Stage 2: Ranking (~100 candidates, <50ms)
- DLRM/DCN-v2 with rich features: user-item cross features, real-time engagement signals
- Feature store provides sub-5ms feature lookups

### Stage 3: Re-ranking (<20ms)
- Business rules: diversity (MMR), freshness, content policy
- Contextual bandit for exploration (10% traffic)

### Stage 4: Policy Layer
- Frequency capping, deduplication, A/B test bucketing

### Feedback Loop
- Log impressions + clicks → join with engagement (watch time, likes)
- Retrain ranking model daily, candidate gen weekly
- Monitor for popularity bias drift

---

## Interview Key Points

- **CF vs content-based?** CF: learns user-item interactions; better for novelty. Content: needs features; good for cold-start items. Hybrid usually best.
- **Why matrix factorization?** Captures latent factors; scalable; low-rank assumption reasonable for most data.
- **How to handle sparse data?** Regularization (L2), factorization (projects to k-dim), sampling negative examples carefully.
- **What is DLRM?** Meta's architecture: embed sparse features, MLP for dense features, dot-product feature interactions, final MLP. Standard for industrial recommendation.
- **How to handle position bias?** Include position as training feature (set to 0 at inference), or use IPS weighting, or jointly learn bias model.
- **Design YouTube recommendations?** 4 stages: candidate gen (two-tower + ANN) → ranking (DLRM) → re-ranking (diversity + explore) → policy. Feature store for real-time signals.
- **What is cold-start?** New user/item has no interactions. Solutions: content features, popularity baseline, ask preferences, exploration.
- **Implicit vs explicit feedback?** Implicit: click ≠ dislike (only know likes). Explicit: ratings. Implicit needs ranking loss (BPR); larger scale.
- **Two-tower model advantage?** Separates user and item representations; enables efficient serving (precompute item embeddings, FAISS search).
- **How to avoid popularity bias?** Diversity penalty during ranking, exploration ε-greedy, calibration (recommend unpopular items with controlled rate).
- **Why NDCG over Precision/Recall?** Precision/Recall treat all positions equally. NDCG: top positions matter more (logarithmic discount).
- **Online vs offline metrics?** Offline: can optimize easily. Online: ground truth (clicks, revenue) but noisy, requires A/B test, longer to iterate.
