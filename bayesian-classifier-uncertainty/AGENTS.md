# AGENTS.md

This file provides context for AI agents working on this codebase.

## Project summary

Bayesian uncertainty quantification for multi-class classifier evaluation. Designed for imbalanced datasets where model predictions are quota-sampled (equal predictions per class) and only a subset of those predictions are human-reviewed.

The goal is to produce honest, decomposed uncertainty estimates for classifier metrics — separating uncertainty from incomplete review (U1) from uncertainty about population-level performance (U2) — and to render these in an interactive HTML report.

See `README.md` for the full generative model and usage instructions.

## File structure

```
uncertainty_analysis.py   # core analysis library
report_template.html      # interactive HTML report template (injected with __DATA_JSON__)
pyproject.toml            # dependencies (numpy, scipy); install with `uv sync`
```

## Generative model (summary)

Review data is quota-sampled within predicted class, so the identified conditional is:

```
Q_k = P(true=j | predicted=k)   modelled as Dirichlet per predicted class k
```

Inference-set confusion uses the exact known predicted-class mix `p_inf = I_k / sum(I_k)`.

Population confusion solves `pi = Q @ p_pop` on the simplex to recover the population predicted-class mix, using known historical prevalences `pi`.

Key assumption: **Q_k is stable between the inference set and the population.** The transport fit residual `||Q p_pop - pi||` is reported as a diagnostic for this assumption.

## Outstanding work

See `TASKS.md` for the current list of outstanding issues and their implementation details.

## Known limitations (do not fix without discussion)

- **Square K×K confusion matrix assumed.** The model requires equal numbers of true and predicted classes. A `# TODO` comment marks the relevant code in `_solve_simplex`. Generalising to rectangular J×K would require changes throughout.
- **Q-stability is an assumption, not verified.** The transport diagnostic flags poor fit but cannot distinguish between a violated stability assumption and a poor simplex solve. Document this clearly if raised.

## Testing

There are no formal tests. The `__main__` block in `uncertainty_analysis.py` runs a simulated 4-class example and prints per-class and aggregate metrics to console. Run with:

```bash
uv run python uncertainty_analysis.py
```

Expected output includes a transport fit residual near zero (simulation uses consistent data) and a generated `uncertainty_report.html`.

## Code style

- NumPy for all numerical operations; avoid Python loops over posterior draws
- `scipy` only for optimisation (to be removed per issue 1 above)
- No other runtime dependencies
- Type hints are not currently used; do not add them without broader refactor
