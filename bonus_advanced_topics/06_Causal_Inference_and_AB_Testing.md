# Causal Inference and A/B Testing

## Overview

**Correlation ≠ Causation**: need experimental design to infer causality.

**A/B Testing**: randomized experiment; compare two versions (control vs treatment); measure impact.

**Causal Inference**: extract causal relationships from observational data (harder than experiments).

---

## Randomized Controlled Experiments (RCT)

### Gold Standard: A/B Testing

**Design**:
1. Randomly assign users to control (old) or treatment (new)
2. Measure outcome (conversion, engagement, revenue)
3. Compare: does treatment differ significantly from control?

**Randomization eliminates bias**: treatment and control groups similar in expectation.

```python
# Simulate A/B test
import numpy as np
from scipy import stats

# Control: baseline conversion rate
n_control = 10000
control_conversions = np.random.binomial(1, p=0.05, size=n_control)

# Treatment: new version (hopefully better)
n_treatment = 10000
treatment_conversions = np.random.binomial(1, p=0.055, size=n_treatment)

# Hypothesis test: is treatment significantly better?
control_rate = control_conversions.mean()
treatment_rate = treatment_conversions.mean()

# Pooled proportion
p_pool = (control_conversions.sum() + treatment_conversions.sum()) / (n_control + n_treatment)
se = np.sqrt(p_pool * (1 - p_pool) * (1/n_control + 1/n_treatment))

# Z-test
z = (treatment_rate - control_rate) / se
p_value = 1 - stats.norm.cdf(z)  # one-tailed

print(f"Control rate: {control_rate:.4f}, Treatment rate: {treatment_rate:.4f}")
print(f"P-value: {p_value:.4f}")
if p_value < 0.05:
    print("Significant difference!")
```

### Sample Size & Power

**Power**: probability of detecting true effect (typically aim for 80% power)

**Sample size formula** (two-sample proportion test):
```
n = 2 * (z_α/2 + z_β)² * p(1-p) / δ²
```

where:
- z_α/2 = critical value for significance level (1.96 for α=0.05)
- z_β = critical value for power (0.84 for 80% power)
- δ = effect size (treatment - control)

```python
from scipy.stats import norm

alpha = 0.05  # significance level
beta = 0.20   # 1 - power
p_control = 0.05
p_treatment = 0.055
delta = p_treatment - p_control

z_alpha = norm.ppf(1 - alpha/2)
z_beta = norm.ppf(1 - beta)
p_pool = (p_control + p_treatment) / 2

n = 2 * (z_alpha + z_beta)**2 * p_pool * (1 - p_pool) / delta**2
print(f"Sample size needed: {n:.0f} per group")
```

### Multiple Testing Correction

**Problem**: run many tests → false positives increase

**Bonferroni correction**: divide significance level by number of tests
```
α_corrected = α / n_tests
```

More conservative; if testing 20 metrics, α = 0.05/20 = 0.0025

**False Discovery Rate (FDR)**: control expected fraction of false positives among all positives
- Less conservative than Bonferroni; better for many tests

```python
from statsmodels.stats.multitest import multipletests

p_values = [...]  # p-values from multiple tests
rejected, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')
```

---

## Observational Causal Inference

Data from real-world without randomization. Much harder; requires assumptions.

### Potential Outcomes Framework

**Counterfactual**: what would have happened if user took different action?

For each unit i:
- Y_i(1) = outcome if treated
- Y_i(0) = outcome if control
- Observed: Y_i = T_i · Y_i(1) + (1 - T_i) · Y_i(0)

**Treatment Effect**: τ_i = Y_i(1) - Y_i(0) (unobservable; only observe one outcome)

**Average Treatment Effect (ATE)**: τ = E[Y(1) - Y(0)]

### Confounding & Bias

**Confounder**: variable affecting both treatment and outcome

Example: user quality (confounder)
- High-quality users more likely to adopt new feature (treatment)
- High-quality users have better outcomes anyway
- Observational comparison biased upward

### Matching: Propensity Score

**Idea**: match treated/control units on propensity to receive treatment.

**Propensity score**: P(T=1 | X) = probability of treatment given features

```python
from sklearn.linear_model import LogisticRegression

# Learn propensity scores
X = features
T = treatment
propensity_model = LogisticRegression()
propensity_scores = propensity_model.fit_predict_proba(X)[:, 1]

# Match: for each treated unit, find similar control unit (by propensity)
from scipy.spatial.distance import cdist

treated_idx = np.where(T == 1)[0]
control_idx = np.where(T == 0)[0]

matches = []
for ti in treated_idx:
    # Find closest control (by propensity score)
    distances = np.abs(propensity_scores[control_idx] - propensity_scores[ti])
    closest_ci = control_idx[np.argmin(distances)]
    matches.append((ti, closest_ci))

# Compare outcomes in matched pairs
ate = np.mean([y[ti] - y[ci] for ti, ci in matches])
```

**Assumption**: no unmeasured confounders (conditional independence)

### Causal Forest

Estimate heterogeneous treatment effects (effect varies by subgroup).

```python
from causalml.inference.tree import CausalTreeRegressor

# Grow trees to maximize treatment effect heterogeneity
causal_forest = CausalTreeRegressor()
te = causal_forest.fit(X_train, T_train, y_train).predict(X_test)
# te[i] = estimated treatment effect for unit i
```

**Use**: identify high-response subgroups for targeting

### Double/Debiased Machine Learning (DML)

Use ML to estimate nuisance parameters (propensity, outcome model); recover causal effect.

```
ATE = E[(T - π(X)) · (Y - m(X)) / (T - π(X))²]
```

where π(X) = propensity, m(X) = outcome model (residuals debiased)

---

## Simpson's Paradox

**Classic error**: trend reverses when data stratified

Example: gender bias in admissions
- Across all departments: more men admitted (aggregate bias)
- Within each department: more women admitted (no bias; men apply to easier depts)

**Lesson**: confounders can flip conclusions; must stratify or control

---

## Experimentation Best Practices

### Experiment Design Checklist

- [ ] **Clear hypothesis**: what are we testing?
- [ ] **Primary metric**: single metric for decision
- [ ] **Sample size**: power calculation done
- [ ] **Randomization**: units independently randomized (avoiding network effects)
- [ ] **Duration**: run long enough (weekly cycles, multiple weeks)
- [ ] **Segmentation**: run only on right population
- [ ] **Guardrail metrics**: ensure no negative side effects
- [ ] **Multiple testing correction**: if many tests
- [ ] **Analysis plan**: written before seeing results (avoid p-hacking)
- [ ] **Stopping rule**: when to stop experiment (not just when p-value < 0.05)

### P-hacking / HARKing (Hypothesizing After Results Known)

**Bad**: look at results, then form hypothesis to confirm
**Good**: pre-specify hypothesis and analysis plan

### Novelty Effect

New treatment excites users → temporarily better metrics → effect fades

**Solution**: run experiment long enough; analyze by cohort (when treatment started)

### Network Effects

Treating user A affects user B (friend); breaks independence

**Solution**: randomize at higher level (cities, regions) if network effects present

---

## Interview Key Points

- **A/B test significance: α vs β?** α = false positive (reject null when true). β = false negative (fail to reject when false). Power = 1 - β.
- **Why randomize?** Eliminates bias; treatment and control similar in expectation; causality.
- **Sample size: how to reduce?** Bigger effect size (easier to detect); increase power tolerance (80% → 70%); smaller significance level costs sample size.
- **Bonferroni vs FDR?** Bonferroni: control false positive rate (strict). FDR: control false discovery rate (lenient); better for many tests.
- **Propensity score: what does it do?** Estimate probability of treatment; matching on it removes confounding (under no unmeasured confounder assumption).
- **Why stratified analysis?** Uncover Simpson's paradox; identify interaction effects; improve efficiency.
- **Novelty effect: how to detect?** Analyze metrics by cohort (when user enrolled); early weeks ≠ later weeks; extrapolate long-term.
- **Guardrail metrics: why?** Detect negative side effects (e.g., quality drops while engagement up); prevent shipping harmful changes.
- **Network effects: why problem?** Randomizing individuals doesn't work; violates independence; bias treatment effect. Solution: cluster randomization.
- **Observational causality: assumptions?** No unmeasured confounders; positivity (all units can be treated/untreated); consistency (no interference).
