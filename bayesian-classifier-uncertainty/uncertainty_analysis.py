"""
Bayesian Classifier Uncertainty Analysis
=========================================
Quantifies and decomposes uncertainty in multi-class classifier metrics across:
  - U1: Sampling uncertainty (incomplete review within inference set)
  - U2: Population uncertainty (inference set != population via pi)

Generative model
----------------
The sampling design is quota-sampling within predicted class. This means the
natural observed conditional is P(true=j | predicted=k), not P(predicted=k | true=j).

  Q_k  ~ Dirichlet(alpha_Q_k)    # P(true=j | predicted=k), one per predicted class
  p_inf = I_k / sum(I_k)         # inference-set predicted-class mix (known exactly)
  C_inf[j,k] = p_inf[k] * Q_k(j) # inference-set confusion matrix (U1)

For population metrics, we solve for the population predicted-class mix p_pop:
  pi = Q @ p_pop  =>  p_pop = argmin_{p in Delta} ||Q p - pi||^2
  C_pop[j,k] = p_pop[k] * Q_k(j) # population confusion matrix (U1 + U2)

The transport fit residual ||Q p_pop - pi|| is reported as a diagnostic:
a large residual indicates the Q-stability assumption is poorly supported.

Key assumption
--------------
Q_k is stable between the inference set and the population — i.e. the model's
error behaviour given a predicted class does not change. Only the mix of
predictions (p) changes between contexts.

Usage
-----
  data    = ReviewData(class_names, review_counts, I_k, pi_counts)
  results = run_analysis(data, n_samples=5000, ci=0.95)
  generate_html_report(results, template_path="report_template.html")
"""

import json
import numpy as np
from pathlib import Path
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Simplex-constrained least squares
# ---------------------------------------------------------------------------

def _solve_simplex(Q, pi):
    """
    Solve  min_{p in Delta^{K-1}}  ||Q @ p - pi||^2

    Q  : (K, K) where Q[j, k] = P(true=j | pred=k)
    pi : (K,)   true class prevalences in the target population
    Returns p : (K,) population predicted-class mix
    """
    K = len(pi)
    res = minimize(
        fun=lambda p: float(np.sum((Q @ p - pi) ** 2)),
        x0=np.ones(K) / K,
        jac=lambda p: 2.0 * Q.T @ (Q @ p - pi),
        method='SLSQP',
        constraints={
            'type': 'eq',
            'fun':  lambda p: p.sum() - 1.0,
            'jac':  lambda p: np.ones(K),
        },
        bounds=[(0.0, 1.0)] * K,
        options={'ftol': 1e-12, 'maxiter': 500},
    )
    p = np.clip(res.x, 0.0, 1.0)
    return p / p.sum()


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

class ReviewData:
    """
    Container for the inputs to the analysis.

    Parameters
    ----------
    class_names : list[str]
        Names of the K classes.

    review_counts : np.ndarray, shape (K, K)
        review_counts[j, k] = number of reviewed items predicted as class k
        whose true label is j. Rows = true class, Cols = predicted class.
        Column k is a draw from Q_k = P(true=j | predicted=k).

    I_k : np.ndarray, shape (K,)
        Total number of predictions of class k in the inference set.
        Defines the inference-set predicted-class mix exactly as I_k / sum(I_k).

    pi_counts : np.ndarray, shape (K,)
        Historical counts of true class k in the population.
        Parameterises the Dirichlet posterior on pi = P(true=k) in population.

    dirichlet_prior_strength : float
        Uninformative prior concentration added to all counts.
        Default 0.5 (Jeffreys prior for Dirichlet).
    """

    def __init__(self, class_names, review_counts, I_k, pi_counts,
                 dirichlet_prior_strength=0.5):
        self.class_names = list(class_names)
        self.K = len(class_names)
        self.review_counts = np.array(review_counts, dtype=float)
        self.I_k = np.array(I_k, dtype=float)
        self.pi_counts = np.array(pi_counts, dtype=float)
        self.prior = dirichlet_prior_strength

        assert self.review_counts.shape == (self.K, self.K), \
            f"review_counts must be ({self.K}, {self.K})"
        assert self.I_k.shape == (self.K,), \
            f"I_k must have shape ({self.K},)"
        assert self.pi_counts.shape == (self.K,), \
            f"pi_counts must have shape ({self.K},)"

    @property
    def alpha_Q(self):
        """
        Dirichlet concentrations for Q_k = P(true=j | pred=k).
        alpha_Q[:, k] = prior + review_counts[:, k]  (column k).
        """
        return self.prior + self.review_counts  # (K, K), col k = alpha for Q_k

    @property
    def alpha_pi(self):
        """Dirichlet concentration for population true-class prevalences."""
        return self.prior + self.pi_counts  # (K,)


# ---------------------------------------------------------------------------
# Posterior sampling
# ---------------------------------------------------------------------------

def sample_posterior(data: ReviewData, n_samples: int = 5000, seed: int = 42):
    """
    Draw samples from the posterior over (Q, pi), derive confusion matrices.

    Returns
    -------
    dict with keys:
        'Q'         : (n_samples, K, K)  Q[s, j, k] = P(true=j | pred=k)
        'pi'        : (n_samples, K)     sampled population prevalences
        'p_inf'     : (K,)               inference-set predicted-class mix (fixed)
        'p_pop'     : (n_samples, K)     population predicted-class mix (solved)
        'C_samp'    : (n_samples, K, K)  inference confusion matrix (U1 only)
        'C_pop'     : (n_samples, K, K)  population confusion matrix (U1 + U2)
        'residuals' : (n_samples,)       transport fit: ||Q p_pop - pi||
    """
    rng = np.random.default_rng(seed)
    K = data.K

    # --- Sample Q: one Dirichlet per predicted class (column k) ---
    # Q[s, j, k] = P(true=j | pred=k)
    Q = np.stack([
        rng.dirichlet(data.alpha_Q[:, k], size=n_samples)
        for k in range(K)
    ], axis=2)  # (n_samples, K, K)

    # --- Inference-set predicted mix: known exactly from I_k ---
    p_inf = data.I_k / data.I_k.sum()  # (K,)

    # --- U1: inference-set confusion ---
    # C_samp[s, j, k] = p_inf[k] * Q[s, j, k]
    C_samp = Q * p_inf[np.newaxis, np.newaxis, :]  # (n_samples, K, K)

    # --- U2: sample pi, solve for population predicted mix ---
    pi_samples = rng.dirichlet(data.alpha_pi, size=n_samples)  # (n_samples, K)

    print(f"Solving population predicted-class mix for {n_samples} posterior draws...", flush=True)
    p_pop = np.array([
        _solve_simplex(Q[s], pi_samples[s])
        for s in range(n_samples)
    ])  # (n_samples, K)

    # Transport fit diagnostic: ||Q p_pop - pi||
    residuals = np.array([
        np.linalg.norm(Q[s] @ p_pop[s] - pi_samples[s])
        for s in range(n_samples)
    ])  # (n_samples,)

    # --- U1 + U2: population confusion ---
    # C_pop[s, j, k] = p_pop[s, k] * Q[s, j, k]
    C_pop = Q * p_pop[:, np.newaxis, :]  # (n_samples, K, K)

    return {
        'Q':         Q,
        'pi':        pi_samples,
        'p_inf':     p_inf,
        'p_pop':     p_pop,
        'C_samp':    C_samp,
        'C_pop':     C_pop,
        'residuals': residuals,
    }


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics_from_cm(C):
    """
    Compute per-class and aggregate metrics from a batch of confusion matrices.

    Parameters
    ----------
    C : np.ndarray, shape (n_samples, K, K)
        Rows = true class, cols = predicted class. Values are proportions.

    Returns
    -------
    dict of metric_name -> np.ndarray, shape (n_samples, K) or (n_samples,)
    """
    eps = 1e-12
    K = C.shape[1]

    TP = C[:, np.arange(K), np.arange(K)]  # (n, K) diagonal
    pred_pos = C.sum(axis=1)                 # (n, K) col sums = P(pred=k)
    true_pos = C.sum(axis=2)                 # (n, K) row sums = P(true=j)
    total    = C.sum(axis=(1, 2))            # (n,)

    FP = pred_pos - TP
    FN = true_pos - TP
    TN = total[:, np.newaxis] - TP - FP - FN

    precision   = TP / (TP + FP + eps)
    recall      = TP / (TP + FN + eps)
    specificity = TN / (TN + FP + eps)
    f1          = 2 * precision * recall / (precision + recall + eps)
    # One-vs-rest accuracy per class
    ovr_accuracy = (TP + TN) / (total[:, np.newaxis] + eps)

    # --- Aggregate scalars ---

    # Overall accuracy
    overall_accuracy = TP.sum(axis=1) / (total + eps)

    # Multiclass MCC (Gorodkin 2004 / Wikipedia formula)
    # MCC = (c*s - Σ_k t_k*p_k) / sqrt((s²-Σ_k p_k²)(s²-Σ_k t_k²))
    c = TP.sum(axis=1)
    s = total
    numerator_mcc = c * s - (true_pos * pred_pos).sum(axis=1)
    denom_mcc = np.sqrt(
        (s ** 2 - (pred_pos ** 2).sum(axis=1)) *
        (s ** 2 - (true_pos ** 2).sum(axis=1))
    )
    mcc = np.where(denom_mcc > eps, numerator_mcc / denom_mcc, 0.0)

    # Cohen's Kappa: κ = (p_o - p_e) / (1 - p_e)
    p_o = overall_accuracy
    p_e = (true_pos * pred_pos).sum(axis=1) / (s ** 2 + eps)
    kappa = np.where(1 - p_e > eps, (p_o - p_e) / (1 - p_e + eps), 0.0)

    # Macro averages (equal class weight)
    macro_precision   = precision.mean(axis=1)
    macro_recall      = recall.mean(axis=1)
    macro_f1          = f1.mean(axis=1)
    macro_specificity = specificity.mean(axis=1)

    # Micro averages (pool counts)
    micro_tp = TP.sum(axis=1)
    micro_fp = FP.sum(axis=1)
    micro_fn = FN.sum(axis=1)
    micro_tn = TN.sum(axis=1)
    micro_precision   = micro_tp / (micro_tp + micro_fp + eps)
    micro_recall      = micro_tp / (micro_tp + micro_fn + eps)
    micro_f1          = 2*micro_precision*micro_recall / (micro_precision+micro_recall+eps)
    micro_specificity = micro_tn / (micro_tn + micro_fp + eps)

    return {
        # Per-class: (n_samples, K)
        'precision':     precision,
        'recall':        recall,
        'specificity':   specificity,
        'f1':            f1,
        'ovr_accuracy':  ovr_accuracy,
        # Aggregate scalars: (n_samples,)
        'overall_accuracy': overall_accuracy,
        'mcc':               mcc,
        'kappa':             kappa,
        'macro_precision':   macro_precision,
        'macro_recall':      macro_recall,
        'macro_f1':          macro_f1,
        'macro_specificity': macro_specificity,
        'micro_precision':   micro_precision,
        'micro_recall':      micro_recall,
        'micro_f1':          micro_f1,
        'micro_specificity': micro_specificity,
    }


def summarise_metrics(metrics, ci=0.95):
    """
    Compute posterior mean, credible intervals, and std for all metrics.

    Returns
    -------
    dict of metric_name -> dict with keys: mean, lower, upper, std, samples
    """
    alpha = (1 - ci) / 2
    summary = {}
    for name, vals in metrics.items():
        summary[name] = {
            'mean':    np.mean(vals, axis=0).tolist(),
            'lower':   np.quantile(vals, alpha, axis=0).tolist(),
            'upper':   np.quantile(vals, 1 - alpha, axis=0).tolist(),
            'std':     np.std(vals, axis=0).tolist(),
            'samples': vals,
        }
    return summary


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(data: ReviewData, n_samples: int = 5000, ci: float = 0.95):
    """
    Run the full Bayesian analysis.

    Returns
    -------
    dict with keys:
        'sample'        : metric summaries — inference set, U1 uncertainty only
        'population'    : metric summaries — population, U1 + U2 uncertainty
        'transport_fit' : diagnostic for Q-stability assumption
        'class_names'   : list of class names
        'ci'            : credible interval level used
        'n_samples'     : number of posterior samples drawn
        'posterior'     : raw posterior samples
    """
    posterior = sample_posterior(data, n_samples=n_samples)

    metrics_samp = compute_metrics_from_cm(posterior['C_samp'])
    metrics_pop  = compute_metrics_from_cm(posterior['C_pop'])

    # Transport fit summary
    residuals = posterior['residuals']
    alpha = (1 - ci) / 2
    transport_fit = {
        'mean':    float(residuals.mean()),
        'lower':   float(np.quantile(residuals, alpha)),
        'upper':   float(np.quantile(residuals, 1 - alpha)),
        'std':     float(residuals.std()),
        'samples': residuals[::5].tolist(),
    }

    mean_residual = transport_fit['mean']
    if mean_residual > 0.05:
        print(f"WARNING: mean transport fit residual = {mean_residual:.4f} (> 0.05). "
              f"The Q-stability assumption may be poorly supported.")
    else:
        print(f"Transport fit residual: {mean_residual:.4f} (good).")

    return {
        'sample':        summarise_metrics(metrics_samp, ci=ci),
        'population':    summarise_metrics(metrics_pop,  ci=ci),
        'transport_fit': transport_fit,
        'class_names':   data.class_names,
        'ci':            ci,
        'n_samples':     n_samples,
        'posterior':     posterior,
    }


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

def generate_html_report(results: dict,
                         template_path: str = "report_template.html",
                         output_path:   str = "uncertainty_report.html"):
    """
    Inject analysis results into the HTML template and write the report.

    The template must contain the placeholder __DATA_JSON__ exactly once.
    """
    per_class_metrics = ['precision', 'recall', 'specificity', 'f1', 'ovr_accuracy']
    aggregate_metrics = [
        'overall_accuracy', 'mcc', 'kappa',
        'macro_f1', 'macro_precision', 'macro_recall', 'macro_specificity',
        'micro_f1', 'micro_precision', 'micro_recall', 'micro_specificity',
    ]

    def serialise_metric(summary, metric):
        s = summary[metric]
        return {
            'mean':    s['mean']  if isinstance(s['mean'],  list) else [s['mean']],
            'lower':   s['lower'] if isinstance(s['lower'], list) else [s['lower']],
            'upper':   s['upper'] if isinstance(s['upper'], list) else [s['upper']],
            'std':     s['std']   if isinstance(s['std'],   list) else [s['std']],
            'samples': s['samples'][::5].tolist() if hasattr(s['samples'], 'tolist')
                       else s['samples'][::5],
        }

    data_payload = {
        'class_names':   results['class_names'],
        'ci':            results['ci'],
        'ci_pct':        int(results['ci'] * 100),
        'sample':        {},
        'population':    {},
        'transport_fit': results['transport_fit'],
    }

    for metric in per_class_metrics + aggregate_metrics:
        data_payload['sample'][metric]     = serialise_metric(results['sample'],     metric)
        data_payload['population'][metric] = serialise_metric(results['population'], metric)

    template = Path(template_path).read_text()
    assert '__DATA_JSON__' in template, \
        "Template must contain the placeholder __DATA_JSON__"

    html = template.replace('__DATA_JSON__', json.dumps(data_payload))
    Path(output_path).write_text(html)
    print(f"Report written to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Example usage / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # REPLACE THIS SECTION WITH YOUR ACTUAL DATA
    # -----------------------------------------------------------------------
    #
    # review_counts[j, k] = number of items predicted as class k
    #                        whose true label is j  (from human review)
    #
    # I_k[k]              = total predictions of class k in inference set
    #
    # pi_counts[k]        = historical count of true class k in population
    #
    # -----------------------------------------------------------------------

    np.random.seed(42)
    K = 4
    class_names = ["Class A", "Class B", "Class C", "Class D"]

    # True prevalences in population (rare classes C and D)
    pi_true = np.array([0.60, 0.30, 0.07, 0.03])

    # Model transition matrix: T[j, k] = P(predicted=k | true=j)
    T_true = np.array([
        [0.85, 0.10, 0.03, 0.02],
        [0.08, 0.82, 0.07, 0.03],
        [0.10, 0.15, 0.68, 0.07],
        [0.05, 0.10, 0.12, 0.73],
    ])

    # Quota-sampled inference set: 500 predictions per class
    I_k = np.array([500, 500, 500, 500], dtype=float)

    # Simulate review counts: m=100 reviews per predicted class k
    # review_counts[:, k] ~ Multinomial(m, P(true=j | pred=k))
    m = 100
    review_counts = np.zeros((K, K))
    for k in range(K):
        # True distribution among items predicted as k: P(true=j | pred=k) via Bayes
        unnorm = pi_true * T_true[:, k]
        Q_k_true = unnorm / unnorm.sum()
        review_counts[:, k] = np.random.multinomial(m, Q_k_true)

    # Historical population counts (e.g. 10,000 labelled historical items)
    pi_counts = pi_true * 10000

    data = ReviewData(
        class_names=class_names,
        review_counts=review_counts,
        I_k=I_k,
        pi_counts=pi_counts,
        dirichlet_prior_strength=0.5,
    )

    print("Review counts (rows=true, cols=predicted):")
    print(review_counts.astype(int))
    print(f"\nI_k (predictions per class): {I_k.astype(int)}")
    print(f"pi_true (population proportions): {pi_true}\n")

    results = run_analysis(data, n_samples=5000, ci=0.95)

    # Console summary
    for level_name, summary in [("SAMPLE (U1 only)",    results['sample']),
                                  ("POPULATION (U1+U2)", results['population'])]:
        print(f"\n=== {level_name} ===")
        for k, name in enumerate(class_names):
            print(f"\n  {name}:")
            for metric in ['precision', 'recall', 'f1', 'specificity']:
                mean  = summary[metric]['mean'][k]
                lower = summary[metric]['lower'][k]
                upper = summary[metric]['upper'][k]
                print(f"    {metric:12s}: {mean:.3f}  95% CI [{lower:.3f}, {upper:.3f}]")

    print("\n=== AGGREGATE (Population) ===")
    for metric in ['overall_accuracy', 'mcc', 'kappa', 'macro_f1', 'micro_f1']:
        s = results['population'][metric]
        mean  = s['mean'][0] if isinstance(s['mean'], list) else s['mean']
        lower = s['lower'][0] if isinstance(s['lower'], list) else s['lower']
        upper = s['upper'][0] if isinstance(s['upper'], list) else s['upper']
        print(f"  {metric:20s}: {mean:.3f}  95% CI [{lower:.3f}, {upper:.3f}]")

    generate_html_report(
        results,
        template_path="report_template.html",
        output_path="/mnt/user-data/outputs/uncertainty_report.html",
    )