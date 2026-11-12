# P4 — paper-ready text (English)

Numbers are from `P4-RESULTS-v1/v2/v3.json` and the PH-3 scalar tables. Each analysis is tagged as preregistered (`PREREG-P4.md`, SHA-256 dbab0f6d…) or post hoc (`DEVIATIONS-P4.md`).

---

## Results text

### Why the action head reads less of the response as denoising proceeds

The quantity of interest is the effective transfer gain $T_k = V_k/R_k$:
- $V_k$ is the energy of the effective velocity difference at denoising step $k$.
- $R_k$ is the energy of the readout difference $\Delta r_k$, the action-expert output after the final adaptive normalization. This is the exact input to the linear action head.
- Both energies are taken over the first five planned positions and the seven exported action dimensions.

$T_k$ falls by a factor of 5.8 from the first to the last step: $\Delta\ln T = -1.75$ [−1.80, −1.71], a condition-balanced mean over 8,673 same-state pairs from 36 tasks.

Because the head is linear, $T_k$ factorizes exactly as $T_k = \rho_k\, g_k\, \nu_k$, where:
- $\rho_k$ is the fraction of readout-difference energy that lies in the head's readable subspace. This subspace is the column space of the 1024×7 weight block, which has numerical rank 7 with singular values 0.64–0.75.
- $g_k$ is the head's gain within that subspace.
- $\nu_k$ is the ratio between the realized update and its linear recomputation; we call it the linear-reference deviation.

The decline is carried almost entirely by $\rho$: it accounts for 98.9% [98.6, 99.2] of $\Delta\ln T$, with the same sign in all ten perturbation conditions. Its share is 0.975–0.997 across perturbation families and anchor positions. By contrast, $g$ is flat ($\Delta\ln g = 0.001$, interval includes 0), and $\nu$ contributes 1.1% (Fig. P4-1a,b).

The response to the perturbation grows 16-fold in energy over the ten steps, almost all of it in the head's null space. Readable energy stays within 7% of its initial value for the first six steps (k = 0–5) and rises mainly in the last three, and the realized update energy follows the readable energy exactly (Fig. P4-1d). This explains why the final step carries a disproportionate share of the update.

The decline is not a generic property of the representation:
- The readable fraction of the clean readout itself falls only 1.6-fold (12.7%→8.0%), whereas that of the response falls 5.7-fold (17.2%→3.0%).
- In a post-hoc counterfactual where both branches are evaluated at the same (clean) denoising state at τ-indices 0/5/9, the response's readable fraction does not fall; it rises from 17.2% to 24.3% (Fig. P4-1c, open markers).

The loss of readability therefore arises only along the native paths, where each branch follows its own state. This is consistent with the perturbed branch's state already absorbing part of the visual effect, which leaves a small readable residual while null-space differences accumulate. We report this as a description, not as a causal identification.

Exceptions are concentrated where responses are small:
- 2.5% of pairs show no decline in $\rho$, mostly under matched global dimming and surface recoloring.
- In the smallest 10% of responses, $\rho$ accounts for 89.5% of the decline and $\nu$ for 11.7%.

### Signed readout direction carries no additional forecast of future updates

**Setup.** We asked whether the signed readout direction at the first step predicts the update that has not yet happened. The information cutoff is after the first denoising update; the target is the update accumulated over the remaining nine steps. All fitting and model selection used only training tasks, and every model had the same tuning budget.

**The simplest baseline is the strongest.** Extrapolating the first update without any fitting ($9\,u_0$) explains 67.7% of the remaining-update energy on held-out tasks (median cosine 0.94; 92.1% at the first query). This beats:
- a magnitude baseline;
- a task-held-out condition mean;
- a fitted linear map from $u_0$ (63.6%).

**Adding the readout does not clear the bar.** Adding the signed readout (its readable coordinates plus principal components of its null-space part) improves on the linear map by only 0.015 [0.013, 0.018]. This is below the preregistered threshold of 0.02, and it does not transfer to any held-out perturbation family (camera −0.027, lighting 0.000, surface −0.006). The readout model remains 0.026 [0.018, 0.035] below the no-fit extrapolation (Fig. P4-2a,b). A post-hoc correction on top of the extrapolation adds −0.0007 [−0.0020, 0.0003].

**Magnitude follows the same pattern.** Readout energy predicts the log-energy of the remaining update less well than the first update's own energy (R² 0.835 vs 0.879), and adds nothing to it (0.879).

**Cross-seed transfer at a fixed state.** At the first query, the observation is identical across noise seeds. Averaging the other seeds' planned command differences predicts the held-out seed's response almost perfectly (EEF 0.931, median cosine 0.977). Replacing that average with a readout-based predictor is worse (−0.031 [−0.040, −0.023]); adding the readout to it changes nothing (+0.0003).

**Later anchors.** Seeds occupy different states at later anchors, so this is not a same-state replication. There, raw command averages fail, and readout-based predictors appear better (+0.20). However, a post-hoc calibrated command average closes this gap to under 0.01. A post-hoc advantage of 0.06 [0.03, 0.10] remains at the first landmark only.

**Conclusion.** Within the information available after the first update, the signed readout direction restates action information that the update already carries; it is not additional knowledge.

---

## Figure captions

**Figure P4-1. The effective transfer gain declines because the perturbation response leaves the head's readable subspace.**

- **(a)** Condition-balanced change, relative to step 0, of the exact factors of the effective transfer gain $T=\rho g\nu$ across the ten denoising steps. Series: readable fraction $\rho$, in-subspace head gain $g$, and linear-reference deviation $\nu$. Population: 8,673 pairs, 36 tasks, 10 persistent visual perturbations, first five planned positions × seven action dimensions.
- **(b)** Step 0→9 change per condition. Blue: $\rho$; orange: $\nu$; the black tick marks $\Delta\ln T$. The shares in the title are task-cluster bootstrap estimates (2,000 replicates).
- **(c)** Readable fraction of the response along the native paths (solid; band is the 95% task bootstrap), of the clean readout itself (dashed), and of the response when both branches share the clean state at τ-indices 0/5/9. The last series (open markers) is a post-hoc shadow-replay counterfactual. The dotted line is the isotropic reference 7/1024.
- **(d)** Geometric-mean energy relative to step 0 of the null-space and readable parts of the response, and of the realized update.

**Figure P4-2. Signed readout direction does not improve forecasts of the not-yet-generated update beyond the first update itself.**

- **(a)** Out-of-fold explained-energy fraction (EEF) for the update accumulated over the remaining nine steps, under 5-fold task-held-out evaluation. Bars are pooled; dots are condition-balanced. All models share the same cutoff (after the first update), the same samples, and the same tuning budget.
- **(b)** Increments with 95% task-bootstrap intervals for:
  - the preregistered primary contrast (signed readout vs a linear map of the first update);
  - the same contrast against the no-fit extrapolation;
  - leave-one-perturbation-family-out tests;
  - a post-hoc readout correction on top of the extrapolation (open diamond).

  The dashed line is the preregistered threshold.
- **(c)** Cross-seed transfer of the planned command response. Only the first query (q0) shares the visited state across seeds. Series:
  - C1: the other seeds' average command response, no fitting;
  - C1s: a post-hoc calibrated version of C1;
  - C2: a readout-based predictor;
  - C3: C1 plus a readout correction.

  The magnitude-only baseline (C0) is negative and not shown.
- **(d)** Per-condition increments of the readout model over the linear-u₀ (filled) and no-fit (open) baselines.

---

## Scope and limitations (for the Discussion)

- **What is measured.** These results describe local generation sensitivity at visited states for one policy (π0.5-LIBERO), 36 tasks and ten photometric/geometric perturbations. They concern planned command prefixes (adapter output), not executed trajectories or task success.
- **Factorization.** The factorization is an exact identity. The empirical content lies in which factor varies, how consistently, and where the exceptions are.
- **The linear-reference deviation $\nu$.** It is small overall but not negligible for the smallest responses. We have not separated its sources (BF16 differencing of the readout, per-branch BF16 head outputs, accumulation order), so we do not attribute it to numerical noise.
- **The shared-state counterfactual.**
  - It is a post-hoc shadow replay at three time points.
  - It shows that loss of readability is path-dependent.
  - It does not establish the mechanism by which the perturbed branch's state absorbs the visual effect.
  - It does not establish that this absorption is protective.
- **Signed-readout forecasting.** The negative result is specific to:
  - the preregistered cutoff (after the first update);
  - linear probes with principal-component features;
  - task- and family-held-out evaluation.

  It does not exclude non-linear decoders or information at other cutoffs. However, a post-hoc check on top of the strongest baseline also found no increment.
- **Cross-seed results at later anchors.** These compare different visited states. The apparent advantage of readout-based predictors there is mostly calibration. The residual advantage at the first landmark is post hoc and unconfirmed.
- **Closed-loop consequences.** These are outside this analysis. In an existing camera × headlight factorial, the camera response direction is largely preserved when the headlight is off (cosine 0.77–0.78). This argues against a headlight-mediated camera effect at the level of generation, but closed-loop attribution would require new rollouts.
