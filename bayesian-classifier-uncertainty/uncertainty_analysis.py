"""
Bayesian Classifier Uncertainty Analysis
=========================================
Quantifies and decomposes uncertainty in multi-class classifier metrics across:
  - U1: Sampling uncertainty (incomplete review of predictions)
  - U2: Population uncertainty (inference set != population via pi)

Generative model:
  pi   ~ Dirichlet(alpha_pi)          # true class prevalences in population
  T_k  ~ Dirichlet(alpha_T_k)         # P(predicted=j | true=k), one row per true class
  C°   = diag(pi) · T                 # population confusion matrix (proportions)

All metrics are deterministic functions of C°, so posteriors are
fully induced by sampling pi and T from their Dirichlet posteriors.

Usage:
  Populate a ReviewData object with your inputs, call run_analysis(),
  then call generate_html_report() pointing at report_template.html.
"""

import json
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Data structures
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
        whose true label is j.
        Rows = true class, Cols = predicted class.

    I_k : np.ndarray, shape (K,)
        Total number of predictions of class k in the inference set.
        Must satisfy I_k[k] >= review_counts[:, k].sum() for all k.

    pi_counts : np.ndarray, shape (K,)
        Historical counts of true class k in the population.
        Used to parameterise the Dirichlet posterior on pi.

    dirichlet_prior_strength : float
        Concentration of the uninformative prior added to all counts.
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
    def alpha_T(self):
        """
        Dirichlet concentration for each row j of T.
        T[j, k] = P(predicted=k | true=j).
        alpha_T[j, :] = prior + review_counts[j, :].
        """
        return self.prior + self.review_counts  # shape (K, K)

    @property
    def alpha_pi(self):
        """Dirichlet concentration for population proportions."""
        return self.prior + self.pi_counts  # shape (K,)


# ---------------------------------------------------------------------------
# Posterior sampling
# ---------------------------------------------------------------------------

def sample_posterior(data: ReviewData, n_samples: int = 5000, seed: int = 42):
    """
    Draw samples from the joint posterior over (pi, T).

    Returns
    -------
    dict with keys:
        'pi'    : (n_samples, K)    — sampled population proportions
        'T'     : (n_samples, K, K) — sampled transition matrices
        'C_pop' : (n_samples, K, K) — population confusion matrix (U1 + U2)
        'C_samp': (n_samples, K, K) — sample confusion matrix (U1 only, pi fixed)
    """
    rng = np.random.default_rng(seed)
    K = data.K

    # Sample T: one Dirichlet per true class (row of T)
    T = np.stack([
        rng.dirichlet(data.alpha_T[j], size=n_samples)
        for j in range(K)
    ], axis=1)  # (n_samples, K, K)

    # Sample pi (for population-level metrics)
    pi_samples = rng.dirichlet(data.alpha_pi, size=n_samples)  # (n_samples, K)

    # Fix pi to its posterior mean (for sample-level / U1-only metrics)
    pi_fixed = data.alpha_pi / data.alpha_pi.sum()
    pi_fixed_rep = np.broadcast_to(pi_fixed, (n_samples, K))

    # C[s, j, k] = pi[s, j] * T[s, j, k]
    C_pop  = pi_samples[:, :, np.newaxis] * T    # (n_samples, K, K)
    C_samp = pi_fixed_rep[:, :, np.newaxis] * T  # (n_samples, K, K)

    return {
        'pi':     pi_samples,
        'T':      T,
        'C_pop':  C_pop,
        'C_samp': C_samp,
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

    TP = C[:, np.arange(K), np.arange(K)]   # (n, K)
    pred_pos = C.sum(axis=1)                  # (n, K) P(predicted=k)
    true_pos = C.sum(axis=2)                  # (n, K) P(true=k)
    total = C.sum(axis=(1, 2))                # (n,)

    FP = pred_pos - TP
    FN = true_pos - TP
    TN = total[:, np.newaxis] - TP - FP - FN

    precision   = TP / (TP + FP + eps)
    recall      = TP / (TP + FN + eps)
    specificity = TN / (TN + FP + eps)
    f1          = 2 * precision * recall / (precision + recall + eps)
    accuracy    = (TP + TN) / (total[:, np.newaxis] + eps)

    # Macro averages (equal weight per class)
    macro_precision   = precision.mean(axis=1)
    macro_recall      = recall.mean(axis=1)
    macro_f1          = f1.mean(axis=1)
    macro_specificity = specificity.mean(axis=1)
    macro_accuracy    = accuracy.mean(axis=1)

    # Micro averages (pool counts across classes)
    micro_tp = TP.sum(axis=1)
    micro_fp = FP.sum(axis=1)
    micro_fn = FN.sum(axis=1)
    micro_tn = TN.sum(axis=1)
    micro_precision   = micro_tp / (micro_tp + micro_fp + eps)
    micro_recall      = micro_tp / (micro_tp + micro_fn + eps)
    micro_f1          = 2*micro_precision*micro_recall / (micro_precision+micro_recall+eps)
    micro_specificity = micro_tn / (micro_tn + micro_fp + eps)

    return {
        # Per-class: shape (n_samples, K)
        'precision':   precision,
        'recall':      recall,
        'specificity': specificity,
        'f1':          f1,
        'accuracy':    accuracy,
        # Macro: shape (n_samples,)
        'macro_precision':   macro_precision,
        'macro_recall':      macro_recall,
        'macro_f1':          macro_f1,
        'macro_specificity': macro_specificity,
        'macro_accuracy':    macro_accuracy,
        # Micro: shape (n_samples,)
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
            'samples': vals,  # kept in memory for report generation; not serialised to disk
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
        'sample'     : metric summaries with pi fixed (U1 uncertainty only)
        'population' : metric summaries with pi sampled (U1 + U2 uncertainty)
        'class_names': list of class names
        'ci'         : credible interval level used
        'n_samples'  : number of posterior samples drawn
        'posterior'  : raw posterior samples (pi, T, C_pop, C_samp)
    """
    posterior = sample_posterior(data, n_samples=n_samples)

    metrics_samp = compute_metrics_from_cm(posterior['C_samp'])
    metrics_pop  = compute_metrics_from_cm(posterior['C_pop'])

    return {
        'sample':      summarise_metrics(metrics_samp, ci=ci),
        'population':  summarise_metrics(metrics_pop,  ci=ci),
        'class_names': data.class_names,
        'ci':          ci,
        'n_samples':   n_samples,
        'posterior':   posterior,
    }


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

def generate_html_report(results: dict,
                         template_path: str = "report_template.html",
                         output_path:   str = "uncertainty_report.html"):
    """
    Inject analysis results into the HTML template and write the report.

    The template must contain the placeholder __DATA_JSON__ exactly once,
    which is replaced with the serialised data payload.

    Parameters
    ----------
    results       : output of run_analysis()
    template_path : path to report_template.html
    output_path   : where to write the final HTML report
    """
    per_class_metrics = ['precision', 'recall', 'specificity', 'f1', 'accuracy']
    aggregate_metrics = ['macro_f1', 'macro_precision', 'macro_recall',
                         'micro_f1', 'micro_precision', 'micro_recall']

    def serialise_metric(summary, metric):
        """Serialise a single metric's summary, downsampling posterior samples 5x."""
        s = summary[metric]
        samples = s['samples']
        downsampled = samples[::5].tolist()
        return {
            'mean':    s['mean']  if isinstance(s['mean'],  list) else [s['mean']],
            'lower':   s['lower'] if isinstance(s['lower'], list) else [s['lower']],
            'upper':   s['upper'] if isinstance(s['upper'], list) else [s['upper']],
            'std':     s['std']   if isinstance(s['std'],   list) else [s['std']],
            'samples': downsampled,
        }

    data_payload = {
        'class_names': results['class_names'],
        'ci':          results['ci'],
        'ci_pct':      int(results['ci'] * 100),
        'sample':      {},
        'population':  {},
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

    # Simulate review counts: m=100 reviews per predicted class
    m = 100
    review_counts = np.zeros((K, K))
    for k in range(K):
        unnorm = pi_true * T_true[:, k]
        p_true_given_pred_k = unnorm / unnorm.sum()
        review_counts[:, k] = np.random.multinomial(m, p_true_given_pred_k)

    # Historical population counts (e.g. from 10,000 labelled historical items)
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
    print(f"pi_true (population proportions): {pi_true}")

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

    generate_html_report(
        results,
        template_path="report_template.html",
        output_path="/mnt/user-data/outputs/uncertainty_report.html",
    )
