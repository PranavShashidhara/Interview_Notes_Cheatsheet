# Probability & Statistics for Data Science / ML / LLMs

## Probability Fundamentals

### Core Rules
- **Addition Rule**: P(A or B) = P(A) + P(B) - P(A and B)
- **Multiplication Rule**: P(A and B) = P(A) * P(B|A)
- **Complement**: P(A') = 1 - P(A)
- **Total Probability**: P(B) = sum over i of P(B|A_i) * P(A_i)

### Conditional Probability
P(A|B) = P(A and B) / P(B)

**Independence**: A and B independent iff P(A and B) = P(A)*P(B)

### Bayes' Theorem
P(A|B) = P(B|A) * P(A) / P(B)

- **Prior**: P(A) — belief before evidence
- **Likelihood**: P(B|A) — how probable is evidence given A
- **Posterior**: P(A|B) — updated belief
- **Marginal**: P(B) — normalizing constant

Application in ML: Naive Bayes classifier, Bayesian hyperparameter optimization (used in your Debt Default project with Optuna).

---

## Probability Distributions

### Discrete
| Distribution | PMF | Mean | Variance | Use Case |
|---|---|---|---|---|
| Bernoulli(p) | p^x(1-p)^(1-x) | p | p(1-p) | Binary outcome |
| Binomial(n,p) | C(n,k)p^k(1-p)^(n-k) | np | np(1-p) | k successes in n trials |
| Poisson(lambda) | e^-lambda * lambda^k / k! | lambda | lambda | Event counts per interval |
| Geometric(p) | (1-p)^(k-1)*p | 1/p | (1-p)/p^2 | Trials until first success |

### Continuous
| Distribution | Key Property | Use Case |
|---|---|---|
| Normal N(mu, sigma^2) | Bell curve, 68-95-99.7 rule | Residuals, features |
| Uniform(a,b) | Flat density | Random initialization |
| Exponential(lambda) | Memoryless | Time between events |
| Beta(alpha, beta) | Defined on [0,1] | Prior on probabilities |
| Gamma(k, theta) | Generalization of Exp | Positive continuous |
| Chi-squared(k) | Sum of k standard normals squared | Goodness of fit |
| t-distribution(k) | Heavy tails vs normal | Small sample inference |
| F-distribution | Ratio of chi-squared | ANOVA, model comparison |

---

## Descriptive Statistics

### Central Tendency
- **Mean**: sum(x_i) / n — sensitive to outliers
- **Median**: middle value — robust to outliers
- **Mode**: most frequent value

### Spread
- **Variance**: E[(X - mu)^2] = E[X^2] - (E[X])^2
- **Std Dev**: sqrt(variance)
- **IQR**: Q3 - Q1 (robust spread)
- **MAD**: Median Absolute Deviation — most robust

### Shape
- **Skewness**: (E[(X-mu)^3]) / sigma^3 — positive = right tail
- **Kurtosis**: (E[(X-mu)^4]) / sigma^4 — excess kurtosis = kurtosis - 3; high = heavy tails

### Covariance and Correlation
- Cov(X,Y) = E[(X-muX)(Y-muY)]
- Pearson r = Cov(X,Y) / (sigma_X * sigma_Y), range [-1, 1]
- Spearman rho = Pearson on ranks — nonparametric
- Correlation does not imply causation

---

## Statistical Inference

### Hypothesis Testing Framework
1. State H0 (null) and H1 (alternative)
2. Choose significance level alpha (typically 0.05)
3. Compute test statistic
4. Compute p-value
5. Reject H0 if p-value < alpha

**p-value**: Probability of observing data at least as extreme as seen, assuming H0 is true. Not the probability H0 is true.

**Type I Error (alpha)**: Rejecting true H0 (false positive)
**Type II Error (beta)**: Failing to reject false H0 (false negative)
**Power (1-beta)**: Probability of correctly rejecting false H0

### Common Tests
| Test | When to Use |
|---|---|
| z-test | Large sample, known variance |
| t-test (one-sample) | Small sample, unknown variance |
| t-test (two-sample) | Compare two means |
| Paired t-test | Before/after same subjects |
| ANOVA | Compare 3+ group means |
| Chi-squared test | Categorical independence / goodness of fit |
| Mann-Whitney U | Non-parametric two-group comparison |
| Kolmogorov-Smirnov | Test distribution equality |

### Confidence Intervals
95% CI for mean: x_bar +/- z_(0.025) * (sigma / sqrt(n))

Wider CI = less certainty. CI does not mean 95% probability parameter is in interval — it means 95% of intervals constructed this way contain the parameter.

---

## Important Theorems

### Central Limit Theorem (CLT)
The sampling distribution of the sample mean approaches Normal as n increases, regardless of the population distribution. Formally: sqrt(n)(X_bar - mu) / sigma -> N(0,1) as n -> infinity.

Critical for: justifying Normal assumptions in large datasets, ML loss landscapes.

### Law of Large Numbers
As n increases, sample mean converges to population mean. Foundation of expected value in ML training.

### Jensen's Inequality
For convex f: f(E[X]) <= E[f(X)]
For concave f: f(E[X]) >= E[f(X)]
Used in: deriving ELBO in VAEs, proving KL divergence is non-negative.

---

## Information Theory (Critical for ML / LLMs)

### Entropy
H(X) = -sum p(x) log p(x)

Measures uncertainty / information content. Maximum entropy = uniform distribution.

### Cross-Entropy
H(p, q) = -sum p(x) log q(x)

Used as loss function in classification: L = -sum y_i log(y_hat_i)

### KL Divergence
KL(p||q) = sum p(x) log(p(x)/q(x))

- Always >= 0 (by Jensen's inequality)
- Not symmetric: KL(p||q) != KL(q||p)
- Used in VAEs, RLHF (policy optimization), regularization

### Mutual Information
I(X;Y) = H(X) - H(X|Y) = KL(p(x,y) || p(x)p(y))

Measures dependence; used in feature selection.

---

## Bayesian Statistics (for ML)

### Bayesian Inference
Posterior proportional to Likelihood * Prior:
p(theta|data) proportional to p(data|theta) * p(theta)

### Maximum Likelihood Estimation (MLE)
theta_MLE = argmax_theta p(data|theta)
For Gaussian: MLE of mean = sample mean; MLE of variance = biased variance (n denominator).

### Maximum A Posteriori (MAP)
theta_MAP = argmax_theta p(theta|data) = argmax_theta [log p(data|theta) + log p(theta)]
Equivalent to MLE + regularization (L2 prior -> Ridge; L1 prior -> Lasso).

### Bayesian Optimization (Optuna / Your Resume)
Used to optimize hyperparameters. Builds a surrogate model (typically Gaussian Process) of the objective function, uses acquisition function (Expected Improvement, UCB) to select next evaluation point. More efficient than grid search for expensive functions.

---

## Statistics for Class Imbalance (Your Resume: ADASYN, Debt Default)

### Why Accuracy Fails
If 99% of data is class 0, a model predicting always class 0 has 99% accuracy but zero utility.

### Better Metrics
- **Precision**: TP / (TP + FP)
- **Recall (Sensitivity)**: TP / (TP + FN)
- **F1**: 2 * Precision * Recall / (Precision + Recall)
- **ROC-AUC**: Area under ROC curve (TPR vs FPR); 0.5 = random, 1.0 = perfect
- **PR-AUC**: Area under Precision-Recall curve; better for severe imbalance

### ADASYN (Adaptive Synthetic Sampling)
Generates synthetic minority samples proportionally in harder-to-learn regions. Differs from SMOTE by focusing generation near decision boundary.

---

## Regression Statistics

### Linear Regression Assumptions
1. Linearity
2. Independence of errors
3. Homoscedasticity (constant variance of errors)
4. Normality of errors
5. No multicollinearity

### Key Metrics
- **R-squared**: 1 - SS_res / SS_tot; proportion of variance explained
- **Adjusted R-squared**: penalizes for extra predictors; R^2_adj = 1 - (1-R^2)*(n-1)/(n-p-1)
- **RMSE**: sqrt(mean((y - y_hat)^2))
- **MAE**: mean(|y - y_hat|)

### Regularization
- **Ridge (L2)**: Loss + lambda * sum(theta_j^2) — shrinks but retains all features
- **Lasso (L1)**: Loss + lambda * sum(|theta_j|) — can zero out features; sparse solutions
- **Elastic Net**: Combines L1 + L2; handles correlated features better than Lasso alone

---

## A/B Testing and Experimentation

### Steps
1. Define metric and minimum detectable effect
2. Calculate sample size (power analysis): n = 2*(z_alpha/2 + z_beta)^2 * sigma^2 / delta^2
3. Randomize users into control/treatment
4. Run for full duration (avoid peeking)
5. Compute p-value / confidence interval
6. Make decision

### Multiple Testing Corrections
- **Bonferroni**: alpha_adj = alpha / m — conservative
- **Benjamini-Hochberg (FDR)**: controls false discovery rate — preferred for large-scale tests

### Common Pitfalls
- Peeking / early stopping inflates Type I error
- Simpson's Paradox: aggregate trend reverses within subgroups
- Novelty effect: initial spike not representative of long-term behavior
- Network effects: SUTVA violation in social settings

---

## Statistics Relevant to LLMs and GenAI

- **Perplexity**: exp(cross-entropy loss); measures how well language model predicts held-out text; lower = better
- **Temperature**: scales logits before softmax; higher temperature = more uniform / diverse; lower = more peaked / deterministic
- **Top-k / Top-p sampling**: probabilistic decoding strategies
- **BLEU, ROUGE**: n-gram overlap metrics for generation quality
- **BERTScore**: semantic similarity using contextual embeddings
- **KL divergence**: used in PPO/RLHF to keep policy close to reference model
- **Calibration**: whether model confidence (probability) matches empirical accuracy; measured with ECE (Expected Calibration Error)

---

## Causal Inference Fundamentals

### Key Concepts
- **Correlation ≠ Causation**: observational data has confounders; randomized experiments (RCTs/A/B tests) are the gold standard
- **Directed Acyclic Graphs (DAGs)**: graphical representation of causal relationships. Nodes = variables, edges = causal effects
- **Confounders**: variables that affect both treatment and outcome. Must control for them to estimate causal effect
- **Colliders**: conditioning on a collider opens a non-causal path (collider bias)
- **Simpson's Paradox**: aggregate trend reverses when data is split by a confounding variable. Example: treatment appears harmful overall but beneficial in every subgroup because of unequal group sizes

### Methods for Observational Data
- **Propensity Score Matching (PSM)**: estimate P(treatment | covariates), match treated/control units with similar propensity scores. Reduces selection bias.
- **Difference-in-Differences (DID)**: compare change in outcome over time between treated and control groups. Assumes parallel trends.
- **Regression Discontinuity Design (RDD)**: exploit a threshold/cutoff in treatment assignment. Compare units just above vs just below.
- **Instrumental Variables (IV)**: use a variable that affects treatment but not outcome directly. Two-stage least squares (2SLS).
- **Backdoor Criterion**: to estimate causal effect of X on Y, block all backdoor paths by conditioning on appropriate variables.

### Interview tip
"How would you measure the effect of X on Y without an A/B test?" → use one of these observational methods depending on data availability.

---

## Bootstrap and Resampling Methods

### Bootstrap Algorithm
1. From dataset of size n, draw n samples **with replacement** → one bootstrap sample
2. Compute statistic of interest (mean, median, model coefficient)
3. Repeat B times (typically B = 1000–10000)
4. Use distribution of B statistics for confidence intervals

### Bootstrap Confidence Intervals
- **Percentile method**: use the 2.5th and 97.5th percentiles of bootstrap distribution for 95% CI
- **BCa (bias-corrected and accelerated)**: adjusts for bias and skewness — preferred in practice

### When to Use Bootstrap
- Distribution of statistic is unknown or hard to derive analytically
- Small sample sizes where CLT may not apply
- Complex statistics (median, ratios, custom metrics)
- Non-parametric — makes no distributional assumptions

### Other Resampling Methods
- **Jackknife**: leave-one-out, estimate bias and variance of an estimator
- **Permutation test**: shuffle labels to build null distribution, compute p-value as fraction of permuted statistics ≥ observed

---

## Advanced A/B Testing

### Variance Reduction — CUPED
- **CUPED (Controlled-experiment Using Pre-Experiment Data)**: use pre-experiment metric values as a covariate to reduce variance
- Adjusted metric: `Y_adj = Y - θ * X`, where X = pre-experiment value, `θ = Cov(X,Y)/Var(X)`
- Can reduce required sample size by 50%+ in practice

### Multi-Armed Bandits
- **Epsilon-greedy**: exploit best arm with probability 1-ε, explore random arm with ε
- **UCB (Upper Confidence Bound)**: pick arm with highest upper confidence bound. Balances exploration/exploitation.
- **Thompson Sampling**: maintain Beta posterior for each arm, sample and pick highest. Bayesian approach, often best in practice.
- When to use bandits vs A/B tests: bandits for continuous optimization (ad serving, recommendation); A/B tests for definitive hypothesis testing

### Network Effects and Interference
- Standard A/B tests assume **SUTVA** (Stable Unit Treatment Value Assumption) — one user's treatment doesn't affect another
- Violated in social networks, marketplaces, ride-sharing
- Solutions: **cluster randomization** (randomize by geographic region), **switchback experiments** (alternate treatment over time periods)

### Sequential Testing
- **Problem with peeking**: checking results before planned sample size inflates false positive rate
- **Group sequential designs**: pre-specified interim analyses with adjusted alpha (O'Brien-Fleming, Pocock boundaries)
- **Always-valid p-values**: anytime-valid inference that controls error rate regardless of when you stop

### Multiple Testing Correction
- **Bonferroni**: divide α by number of tests. Conservative.
- **Benjamini-Hochberg (FDR)**: controls false discovery rate. Less conservative, preferred when many tests.
- **Holm-Bonferroni**: step-down procedure, uniformly more powerful than Bonferroni

---

## Statistical vs Practical Significance

- **Statistical significance**: p-value < α (typically 0.05). The effect is unlikely due to chance.
- **Practical significance**: the effect size is large enough to matter for the business.
- A/B test with 10M users can detect a 0.01% lift with p < 0.05 — statistically significant but practically meaningless.

### Effect Size Measures
- **Cohen's d** = (μ₁ - μ₂) / σ_pooled → Small: 0.2, Medium: 0.5, Large: 0.8
- **Cohen's h** for proportions: h = 2·arcsin(√p₁) - 2·arcsin(√p₂)
- **Relative lift**: (treatment - control) / control × 100%
- **Minimum Detectable Effect (MDE)**: smallest effect size your test is powered to detect. Always define MDE before running a test.

### P-Value Misconceptions (Common Interview Trap)
- "P-value is the probability the null hypothesis is true" → **Wrong.** It's P(data | H₀), not P(H₀ | data).
- "P = 0.03 means 3% chance the result is due to chance" → **Wrong.** It means: if H₀ were true, we'd see data this extreme 3% of the time.
- "P > 0.05 means no effect" → **Wrong.** Insufficient evidence ≠ evidence of absence.
- "P = 0.05 is a universal threshold" → **Wrong.** It's arbitrary; some fields use 0.01 or 0.001.

---

## Survival Analysis Basics

### Core Concepts
- **Survival function S(t)** = P(T > t): probability of surviving past time t
- **Hazard function h(t)** = instantaneous failure rate at time t, given survival to t
- **Censoring**: when we don't observe the event (user still active at study end = right-censored)

### Key Methods
- **Kaplan-Meier estimator**: non-parametric estimate of S(t). Steps down at each observed event time. Handles censoring.
- **Log-rank test**: compare survival curves between two groups (e.g., treatment vs control)
- **Cox Proportional Hazards**: semi-parametric model. `h(t|X) = h₀(t) · exp(β·X)`. Estimates effect of covariates on hazard without specifying baseline hazard.

### ML Applications
- Customer churn prediction (time to churn)
- User retention modeling
- Equipment failure prediction (predictive maintenance)
- Clinical trial outcomes

### Interview tip
"How would you model time-to-event data where some users haven't churned yet?" → survival analysis with censoring, not standard classification.
