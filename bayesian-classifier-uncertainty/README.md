# Bayesian Classifier Uncertainty Analysis

Estimates and decomposes uncertainty in multi-class classifier metrics for imbalanced datasets where predictions are quota-sampled and only a subset of predictions are human-reviewed.

## Uncertainty sources

- **U1 (sampling):** incomplete review of predictions within the inference set
- **U2 (population):** population prevalence differing from the inference set distribution

These are tracked separately, enabling statements like: *"F1 is 0.72 [0.68, 0.76] in our sample, but 0.71 [0.65, 0.77] in the population."*

## Method

Bayesian generative model with conjugate Dirichlet posteriors — no MCMC required.

Because review data is quota-sampled within predicted class, the identified conditional is `Q_k = P(true=j | predicted=k)`, not `P(predicted=k | true=j)`. The model is:

```
Q_k  ~ Dirichlet(prior + review_counts[:, k])   # one per predicted class
p_inf = I_k / sum(I_k)                           # inference-set predicted mix (exact)
C_inf[j,k] = p_inf[k] * Q_k(j)                  # inference confusion matrix (U1)
```

For population metrics, the population predicted-class mix `p_pop` is recovered by solving the simplex-constrained linear system implied by the known population prevalences `pi`:

```
pi = Q @ p_pop   =>   p_pop = argmin_{p in Delta} ||Q p - pi||^2
C_pop[j,k] = p_pop[k] * Q_k(j)                  # population confusion matrix (U1+U2)
```

This requires one assumption: **Q_k is stable between the inference set and the population** — i.e. the model's error behaviour given a predicted class does not change. A transport fit diagnostic (`||Q p_pop - pi||`) is reported to flag when this assumption is poorly supported.

All metrics are derived from posterior samples of the confusion matrix. Metrics reported: precision, recall, F1, specificity, one-vs-rest accuracy (per class); overall accuracy, MCC, Cohen's Kappa, macro and micro averages (aggregate).

## Usage

```python
from uncertainty_analysis import ReviewData, run_analysis, generate_html_report

data = ReviewData(
    class_names=["A", "B", "C"],
    review_counts=review_counts,  # (K, K) array: rows=true, cols=predicted
    I_k=I_k,                      # total predictions per class in inference set
    pi_counts=pi_counts,          # historical population counts per class
)

results = run_analysis(data, n_samples=5000, ci=0.95)
generate_html_report(results, template_path="report_template.html")
```

## Inputs

| Parameter | Shape | Description |
|---|---|---|
| `review_counts[j, k]` | (K, K) | Reviewed items predicted as k with true label j |
| `I_k` | (K,) | Total predictions of class k in inference set |
| `pi_counts` | (K,) | Historical population counts per class |

## Files

- `uncertainty_analysis.py` — core analysis library
- `report_template.html` — interactive HTML report template