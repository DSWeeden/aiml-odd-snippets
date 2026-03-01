# Bayesian Classifier Uncertainty Analysis

Estimates and decomposes uncertainty in multi-class classifier metrics for imbalanced datasets where predictions are quota-sampled and only a subset of predictions are human-reviewed.

## Uncertainty sources

- **U1 (sampling):** incomplete review of predictions within the inference set
- **U2 (population):** inference set distribution differing from the true population

These are tracked separately, enabling statements like: *"F1 is 0.72 [0.68, 0.76] in our sample, but 0.71 [0.65, 0.77] in the population."*

## Method

Bayesian generative model with conjugate Dirichlet posteriors — no MCMC required:

```
pi  ~ Dirichlet(alpha_pi)     # population class prevalences
T_k ~ Dirichlet(alpha_T_k)    # P(predicted=k | true=j), per true class
C°  = diag(pi) · T            # population confusion matrix
```

All metrics (precision, recall, F1, specificity, accuracy; macro + micro) are derived from posterior samples of `C°`.

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