# Bonus Advanced Topics

This folder contains advanced and supplementary topics for comprehensive interview preparation.

## Contents

### 01_Time_Series_and_Forecasting.md
Deep dive into time series analysis and forecasting methods.
- Classical methods: ARIMA, Prophet, exponential smoothing
- Deep learning: LSTM, Transformers, TCN
- Evaluation metrics and stationarity
- Best practices: walk-forward validation, handling seasonality

**When to study**: For finance, operations, IoT, or any time-dependent data roles.

---

### 02_Graph_Neural_Networks.md
Comprehensive guide to neural networks on graph-structured data.
- Graph basics and representations
- GCN, GraphSAGE, GAT, GIN architectures
- Message passing, pooling, and readout
- Knowledge graphs and embeddings
- Applications: social networks, molecules, recommendations

**When to study**: For roles involving networks, social graphs, or molecular data.

---

### 03_Reinforcement_Learning.md
Foundations of reinforcement learning and decision-making.
- Markov Decision Processes (MDPs)
- Value functions: V(s) and Q(s,a)
- Model-free methods: Q-learning, DQN, policy gradients
- Actor-Critic and PPO algorithms
- Exploration vs exploitation, reward shaping

**When to study**: For robotics, game AI, autonomous systems, or optimization roles.

---

### 04_Anomaly_Detection.md
Methods for identifying outliers and unusual patterns.
- Statistical methods: Z-score, isolation forest, LOF, one-class SVM
- Reconstruction-based: autoencoders, VAE, PCA
- Time series anomalies: decomposition, ARIMA residuals
- Evaluation: ROC-AUC, precision-recall, threshold selection
- Practical considerations: imbalance, concept drift, multiple types

**When to study**: For fraud detection, security, monitoring, or quality control roles.

---

### 05_Clustering_and_Unsupervised_Learning.md
Partition and explore unlabeled data.
- K-means, hierarchical clustering, DBSCAN, HDBSCAN
- Gaussian Mixture Models (GMM)
- Choosing k: elbow method, silhouette score
- Dimensionality reduction: PCA, t-SNE, UMAP
- Evaluation metrics: silhouette, Davies-Bouldin, Calinski-Harabasz

**When to study**: For customer segmentation, exploratory analysis, or unsupervised roles.

---

### 06_Causal_Inference_and_AB_Testing.md
From correlation to causation: experiments and observational analysis.
- A/B testing fundamentals: randomization, sample size, power
- Multiple testing correction: Bonferroni, FDR
- Observational causal inference: propensity scores, matching
- Causal forests for heterogeneous effects
- Experimentation best practices: novelty effect, network effects

**When to study**: For product analytics, experimentation, or growth roles.

---

### 07_Interview_Strategies_and_Communication.md
How to ace ML/data interviews: communication, problem-solving, and avoiding pitfalls.
- Problem-solving framework: FAST (Frame, Approach, Suggest, Trade-offs)
- Communication tips: clarity, avoiding jargon, asking for guidance
- Common mistakes: data leakage, ignoring imbalance, overfitting
- Handling difficult moments: "I don't know", time pressure, disagreement
- Domain-specific tips: recommendations, NLP, CV, time series
- Pre-interview checklist and mindset

**When to study**: Before every interview! Especially when you're getting good technical answers but want to improve communication.

---

## How to Use This Folder

### Focused Preparation
If you have a specific focus area:
1. Study the corresponding file in-depth
2. Solve practice problems in that domain
3. Explain concepts to a friend
4. Mock interview with domain-specific questions

### Comprehensive Preparation
Go through all files in order over 2-3 weeks:
1. Start with easier topics (Time Series, Clustering)
2. Move to moderate difficulty (GNNs, Anomaly Detection)
3. Study hard topics (RL, Causal Inference)
4. Apply strategies from Interview_Strategies throughout

### Just-in-Time Learning
Look up a topic 1-2 days before interview on that domain.

---

## Relationship to Main Cheatsheets

**Main cheatsheets** (root folder) cover:
- Core ML fundamentals (06_LLMs_and_GenAI, 05_Deep_Learning_and_CV, 12_NLP)
- Essential tools (09_Spark_PyTorch_ONNX_MLflow, 14_Distributed_Training)
- System design (11_Software_Engineering_System_Design)
- Cloud/deployment (15_AWS_and_Cloud_ML)

**Bonus topics** (this folder) dive deeper into:
- Specialized domains (time series, graphs)
- Advanced concepts (RL, causal inference)
- Interview meta-skills (communication, strategy)

**Top 3 files from main folder** (new):
- 16_Recommendation_Systems: high-frequency interview topic
- 17_Feature_Engineering_and_Selection: foundational for all ML
- 18_Explainability_and_Interpretability: increasingly asked

---

## Suggested Study Time

| Topic | Time | Difficulty |
|-------|------|------------|
| Time Series | 1-2 hours | Medium |
| GNNs | 1.5-2 hours | Hard |
| RL | 2-3 hours | Hard |
| Anomaly Detection | 1 hour | Easy |
| Clustering | 1-1.5 hours | Easy |
| Causal/A/B Testing | 1.5-2 hours | Medium |
| Interview Strategies | 0.5 hours | Easy |
| **Total** | **~9-12 hours** | |

---

## Tips for Success

1. **Don't memorize**: Understand intuition; memorizing formulas without understanding hurts
2. **Practice code**: For each topic, write minimal example code
3. **Explain to others**: Teaching a concept is the best way to learn
4. **Connect to applications**: Link each topic to real-world problems
5. **Mix and match**: Combine concepts (e.g., anomaly detection + time series)
6. **Revisit weak areas**: Spend extra time on topics where you get stuck
7. **Practice interviews**: Mock interviews > reading

---

## Additional Resources

- **Time Series**: "Forecasting: Principles and Practice" (Hyndman & Athanasopoulos)
- **GNNs**: "A Comprehensive Survey on Graph Neural Networks" (Wu et al.)
- **RL**: "Sutton & Barto: Reinforcement Learning"
- **Causal Inference**: "Causal Inference: The Mixtape" (Cunningham)
- **Anomaly Detection**: Scikit-learn docs, Kaggle competitions

---

Good luck with your interview preparation!
