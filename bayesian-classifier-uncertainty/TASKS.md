# TASKS.md

Outstanding issues in priority order. Update this file as work is completed.

---

## 1. Replace SLSQP simplex solve with fast vectorised projection (HIGH)

**File:** `uncertainty_analysis.py`, function `_solve_simplex`

**Problem:** The current implementation calls `scipy.optimize.minimize` (SLSQP) once per posterior draw in a Python loop. At K=100 with n_samples=5000 this will be unacceptably slow and may have convergence failures.

**Fix:** Replace with a two-step approach vectorised across all draws simultaneously:

1. Solve unconstrained least squares: `p = (QᵀQ)⁻¹Qᵀπ` — closed form, batchable across all draws with `np.linalg.lstsq`
2. Project each solution onto the probability simplex using the Duchi et al. (2008) O(K log K) algorithm:
   - Sort `p` descending
   - Find `ρ = max{j : p_j - (1/j)(sum(p_1..p_j) - 1) > 0}`
   - Compute `θ = (sum(p_1..p_ρ) - 1) / ρ`
   - Return `max(p - θ, 0)`

The entire operation should be vectorised across all n_samples draws using numpy — no Python loop. Once done, remove the progress print statement in `sample_posterior` and remove `scipy` from `pyproject.toml`.

---

## 2. Replace hard-coded residual threshold with relative measure (MEDIUM)

**File:** `uncertainty_analysis.py`, function `run_analysis`; `report_template.html`, transport fit card

**Problem:** The warning threshold `mean_residual > 0.05` is an absolute L2 norm, which does not scale with K. For K=100, norms are naturally larger and this threshold will produce spurious warnings.

**Fix:** Switch to a relative residual: `||Q p_pop - pi||₂ / ||pi||₂`. Changes required:
- Compute and store both absolute and relative residuals in `sample_posterior`
- Add a `residual_threshold` parameter to `run_analysis` (default `0.1`) applied to the relative measure
- Update the `transport_fit` dict to include `relative_mean`, `relative_lower`, `relative_upper`
- Update the transport fit card in `report_template.html` to display the relative residual alongside the absolute one and use the relative measure for the colour-coded fit badge

---

## 3. Strengthen U1/U2 decomposition language (LOW)

**File:** `report_template.html`, decomposition panel note

**Problem:** The note says `Total std ≈ sqrt(U1² + U2²) — heuristic; covariance term omitted.` This understates the issue: where U1 and U2 are positively correlated — likely for rare classes — the formula will *overestimate* total uncertainty, not just be imprecise.

**Fix:** Replace the note with:
> `Total std ≈ √(U1² + U2²)` is a heuristic that omits the covariance term `2·Cov(U1, U2)`. For rare classes, U1 and U2 tend to be positively correlated, so this formula will overestimate total uncertainty. For an exact decomposition, compute variance directly from joint posterior draws.
