# Explainability and Interpretability

## Overview

**Interpretability**: model decisions understandable to humans (inherent property)
**Explainability**: explaining model predictions post-hoc (adding transparency)

**Why it matters**: regulatory (GDPR, HIPAA), trust, debugging, feature discovery, bias detection.

---

## Inherently Interpretable Models

### Linear Models
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

**Interpretability**: coefficient βᵢ = change in output per unit change in xᵢ

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, y)
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature}: {coef:.4f}")
```

**Pros**: transparent, fast, regulatory-friendly
**Cons**: limited capacity; assumes linearity

### Decision Trees
Splits on features; path from root to leaf explains prediction.

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

model = DecisionTreeClassifier(max_depth=5)
model.fit(X, y)
plot_tree(model, feature_names=X.columns)
```

**Pros**: mirrors human decision logic
**Cons**: unstable (small data changes → big tree changes); prone to overfitting if deep

### Rule-Based Models
Explicit if-then rules; understandable without training.

```python
# Example: lending
if age > 25 and income > 50k and credit_score > 700:
    approve_loan()
```

**Pros**: fully transparent, fast inference, regulatory-friendly
**Cons**: hard to learn optimal rules; limited by rule complexity

---

## Post-hoc Explanations (Black Box Models)

### Feature Importance (Tree-based)

#### Gini/Entropy Importance
Measures reduction in impurity when feature is split.

```python
model = RandomForestClassifier()
model.fit(X, y)
importances = model.feature_importances_
for feature, imp in sorted(zip(X.columns, importances)):
    print(f"{feature}: {imp:.4f}")
```

**Pros**: fast (computed during training); easy to understand
**Cons**: biased toward high-cardinality features; doesn't account for feature correlations

#### Permutation Importance
Drop-in accuracy when feature values are shuffled (broken).

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_val, y_val, n_repeats=10)
for feature, imp in sorted(zip(X.columns, result.importances_mean)):
    print(f"{feature}: {imp:.4f}")
```

**Interpretation**: larger drop in accuracy → more important feature
**Pros**: model-agnostic (works for any model); more reliable
**Cons**: expensive (requires retraining on shuffled data); correlated features → shared importance

---

## SHAP (SHapley Additive exPlanations)

Game-theoretic approach: feature contribution to prediction.

### SHAP Values

Each feature's contribution: prediction = base_value + sum(SHAP_values)

```python
import shap

# For tree-based models (fast)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# For any model (slow but general)
explainer = shap.KernelExplainer(model.predict, X_sample)
shap_values = explainer.shap_values(X)

# Visualizations
shap.summary_plot(shap_values, X)  # feature importance + direction
shap.force_plot(explainer.expected_value, shap_values[0], X.iloc[0])  # single prediction
shap.dependence_plot("age", shap_values, X)  # age vs SHAP value
```

**Interpretation**:
- Positive SHAP value: feature pushes prediction up
- Negative SHAP value: feature pushes prediction down
- Larger magnitude: larger impact

**Pros**: theoretically sound (game theory); accounts for interactions; global + local explanations
**Cons**: computationally expensive; requires samples for KernelExplainer

### SHAP vs Permutation Importance
- **SHAP**: shows direction of effect (helps debug model)
- **Permutation**: magnitude of drop in accuracy (interpretable business metric)

---

## LIME (Local Interpretable Model-agnostic Explanations)

Approximate complex model with simple, local linear model.

```python
from lime.tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(X_train, feature_names=X.columns)
exp = explainer.explain_instance(x_instance, model.predict_proba)
exp.show_in_notebook()
```

**How it works**:
1. Perturb instance: generate similar instances (small random changes)
2. Get predictions for perturbed instances
3. Fit weighted linear model (weight by similarity to original)
4. Linear coefficients = local importance

**Pros**: model-agnostic; produces interpretable local approximations
**Cons**: local only (not global); approximation quality depends on linear fit

**LIME vs SHAP**:
- LIME: local explanations; linear approximation
- SHAP: global + local; theoretically grounded

---

## Attention Visualization (Neural Networks)

Attention weights show which parts of input the model attends to.

```python
# Transformer model
outputs = model(input_ids, output_attentions=True)
attention_heads = outputs[-1]  # attention matrices

# Visualize: which tokens attend to which
import matplotlib.pyplot as plt
plt.imshow(attention_heads[0, 0].detach().numpy())  # layer 0, head 0
plt.colorbar()
plt.show()
```

**Use Cases**:
- NLP: which words affect prediction (sentiment, QA)
- Vision: which image regions matter (saliency maps)

**Limitations**: attention ≠ importance (attends doesn't mean causally relevant)

---

## Gradient-based Saliency Maps

Compute gradient of output w.r.t. input features.

```python
import torch

x = torch.tensor(x_input, requires_grad=True)
output = model(x)
output.backward()
gradients = x.grad  # ∂output/∂input
```

**Interpretation**: large gradient → small change in input → large change in output → important

**Variants**:
- **Saliency**: raw gradients
- **Integrated Gradients**: accumulate gradients along path from baseline to input (more stable)
- **Grad-CAM**: visualization of class-activation maps (for CNNs)

```python
# Integrated Gradients
ig = IntegratedGradients(model)
attributions = ig.attribute(x, baselines=baseline, target=target_class)
```

**Pros**: fast (single backward pass)
**Cons**: doesn't capture second-order effects; can be noisy

---

## Concept-based Explanations

### TCAV (Testing with Concept Activation Vectors)

Learn vector representing high-level concept (e.g., "cat-ness", "gender"), then measure importance.

```
1. Collect images with/without concept
2. Learn classifier to separate concept examples from rest
3. Extract concept vector (normal to decision boundary)
4. Measure model sensitivity to concept vector
```

**Use**: understand if model uses human-interpretable concepts
**Limitation**: requires labeled concept examples

---

## Model Debugging

### Partial Dependence Plot (PDP)
Shows marginal effect of feature on prediction (average over other features).

```python
from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(model, X, ['age', 'income'])
plt.show()
```

**Interpretation**: expected output as feature varies (others held at average)
**Use**: debug non-intuitive feature relationships

### Accumulated Local Effects (ALE)
Similar to PDP, but more interpretable for correlated features.

```python
from pyaleplot import ale

ale(model, X, features=['age', 'income'])
```

### Confusion Matrix + ROC Curve
Classic tools for classification debugging.

```python
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)
fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
plt.plot(fpr, tpr, label=f'AUC={auc(fpr, tpr):.3f}')
```

---

## Fairness & Bias Detection

### Demographic Parity
Prediction rate same across groups.

```python
# Should equal for all groups
P(Y_pred=1 | Group=A) == P(Y_pred=1 | Group=B)
```

### Equalized Odds
True positive rate + false positive rate same across groups.

```python
# Both should be equal across groups
P(Y_pred=1 | Y=1, Group=A) == P(Y_pred=1 | Y=1, Group=B)  # TPR
P(Y_pred=1 | Y=0, Group=A) == P(Y_pred=1 | Y=0, Group=B)  # FPR
```

### Bias Audit
```python
from fairness.metrics import disparate_impact

# Disparate impact ratio: 80/20 rule (hiring)
ratio = P(Y_pred=1 | Group_B) / P(Y_pred=1 | Group_A)
assert ratio >= 0.8, "Disparate impact detected"
```

---

## Practical Workflow

### Black Box Model
1. **Feature Importance**: permutation or SHAP (global picture)
2. **LIME**: explain individual predictions (local)
3. **SHAP dependence**: understand feature-output relationships
4. **Fairness audit**: check for bias across groups

### Interpretable Model
1. Use linear/tree models when possible
2. If black box needed: add post-hoc explanations (SHAP, LIME)
3. Monitor feature importance over time (data drift, concept drift)

---

## Interview Key Points

- **Interpretability vs Explainability?** Interpretability: inherent (e.g., linear models). Explainability: added post-hoc (SHAP, LIME for black boxes).
- **When to use interpretable models?** High-stakes decisions (medical, legal, lending); regulatory requirements; need to understand feature relationships.
- **SHAP vs LIME?** SHAP: global + local, theoretically sound, expensive. LIME: local only, fast, linear approximation.
- **Permutation importance bias?** Correlated features share importance; high-cardinality features ranked high. Use SHAP for more nuanced view.
- **Attention in transformers = importance?** No. Attention says "which tokens matter for sequence"; doesn't mean causally important. Verify with gradients/perturbation.
- **How to explain neural network?** Gradient-based (saliency, Integrated Gradients), attention, SHAP, LIME.
- **Feature importance: global vs local?** Global (SHAP summary, permutation): feature matters on average. Local (LIME, force_plot): matters for this instance.
- **How to detect if model has bias?** Demographic parity/equalized odds checks; SHAP values by group; fairness audits; confusion matrices per group.
- **Why explain?** Debug errors, regulatory compliance, model improvement, stakeholder trust.
