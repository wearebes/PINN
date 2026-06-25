# Positive-Only Numeric And D4 Symmetry Comparison Design

Date: 2026-06-25
Scope: Part 2 comparison package for the newly finished data-volume and `sdf+nonsdf` model results in `/Users/jcy/research/PINN`.

## 1. Goal

Produce a comparison package that answers three questions clearly:

1. Which model is best on the positive-only numeric evaluation the user currently cares about.
2. Whether increasing data volume (`main`, `2x`, `3x`) gives a stable numeric improvement.
3. Whether mixing `sdf+nonsdf` changes numeric performance and D4 symmetry behavior in a meaningful way.

The package is for comparison and reporting. It is not a training plan and it does not introduce new metrics beyond the repo's existing result surfaces unless a later implementation step proves they are already computed and consistent.

## 2. Current Repo Facts Used By This Design

The repo currently exposes these comparison objects as the defensible default set:

- `baseline_256_hgradient`
- `dcts_main_wd0`
- `dcts_main_2x`
- `dcts_main_3x`
- `dcts_main_sdfnonsdf`
- `dcts_main_2x_sdfnonsdf`
- `dcts_main_3x_sdfnonsdf`

Evidence for this set exists in:

- `out/256/baseline_256_hgradient.pt`
- `out/7367/dcts_main_wd0.pt`
- `out/dcts_volume_experiment/dcts_main_2x.pt`
- `out/dcts_volume_experiment/dcts_main_3x.pt`
- `out/dcts_volume_experiment/dcts_main_sdfnonsdf.pt`
- `out/dcts_volume_experiment/dcts_main_2x_sdfnonsdf.pt`
- `out/dcts_volume_experiment/dcts_main_3x_sdfnonsdf.pt`

The user originally described `baseline + 6 models` and also said `8 indicators`. Because no eighth model object was confirmed during brainstorming, this design fixes the object set at `1 baseline + 6 models = 7 objects`. The phrase `8 indicators` is treated as a reporting-layer requirement rather than an object-count requirement.

Existing numeric evaluation evidence already lives in:

- `out/7367/test2_resolution_sweep_metrics.csv`

Existing symmetry methodology evidence already lives in:

- `tem/eta_bin_breakdown/plot_symmetry_comparison.py`

This design reuses those surfaces rather than inventing a new comparison contract.

## 3. Hard Comparison Contract

### 3.1 Positive-only rule

Primary numeric comparisons must use positive-only samples. Negative-branch results are excluded from the main package.

This means:

- no negative-only plots in the main comparison output
- no positive-vs-negative twin overlays in the main comparison output
- if negative-side diagnostics are preserved at all, they belong in appendix or implementation-side scratch outputs, not the primary deliverable

### 3.2 Symmetry rule

The symmetry comparison must only use the D4 symmetry MSE view that the user explicitly requested.

This means:

- the main symmetry figure uses `D4 symmetry MSE` only
- `mean violation` and `max violation` are not primary comparison metrics
- they may exist as implementation diagnostics, but they are outside the approved main design

### 3.3 Baseline rule

The baseline is fixed at resolution-256 `nngradient`, represented in-repo by `baseline_256_hgradient`.

It must appear in the comparison package, but its role differs by section:

- in numeric comparison panels, it is a first-class comparison object
- in the symmetry panel, it is a labeled reference baseline, not a peer with the same D4-training semantics as the part2 models

## 4. Final Deliverable Shape

The approved package is:

- `2 groups of figures`
- `1 summary table`

This is intentionally not a single dense all-in-one chart. Numeric behavior and symmetry behavior are separated because they answer different questions and use different interpretation rules.

### 4.1 Group A: Positive-only numeric comparison

This group answers which model is numerically better on the positive-only evaluation target.

Primary metrics:

- `overall MSE`
- `overall MAE`
- `circle MSE`
- `ellipse MSE`

These metrics were chosen because they are the smallest stable set that still separates:

- overall quality
- geometry-specific behavior
- direct comparison against the baseline

### 4.2 Group B: D4 symmetry-only comparison

This group answers how the models compare under the approved D4 symmetry MSE interpretation.

Primary metric:

- `D4 symmetry MSE`

No additional symmetry diagnostics belong in the primary conclusion layer.

### 4.3 Summary table

The final table provides one row per comparison object and serves as the report-reading entry point after the figures.

Required columns:

- `model`
- `overall MSE`
- `overall MAE`
- `circle MSE`
- `ellipse MSE`
- `D4 symmetry MSE`
- `vs baseline MSE delta`

## 5. Object Ordering And Grouping

All object-based plots must use the same left-to-right order:

1. `baseline_256_hgradient`
2. `dcts_main_wd0`
3. `dcts_main_2x`
4. `dcts_main_3x`
5. `dcts_main_sdfnonsdf`
6. `dcts_main_2x_sdfnonsdf`
7. `dcts_main_3x_sdfnonsdf`

This order is fixed for interpretation, not aesthetics.

It deliberately exposes two chains:

- data-volume chain: `main -> 2x -> 3x`
- field-mode chain: `main_sdfnonsdf -> 2x_sdfnonsdf -> 3x_sdfnonsdf`

The baseline stays first so every reader can anchor all subsequent deltas against the same reference.

## 6. Figure Layout

Use a simple model-object x-axis for every main panel. Do not use training step, resolution, or eta-bin as the main x-axis for this package, because this deliverable is a discrete model-version comparison, not a sweep study.

Recommended panel layout:

- Row 1: `overall MSE`, `overall MAE`
- Row 2: `circle MSE`, `ellipse MSE`
- Row 3: `D4 symmetry MSE`

Followed by:

- `summary table`

Interpretation order should be:

1. overall quality
2. geometry-specific differences
3. symmetry behavior
4. consolidated table conclusion

## 7. Visual Encoding Rules

The styling must help readers see experiment families immediately.

Required color logic:

- baseline uses a neutral color
- `main / 2x / 3x` use one shared hue family with ordered intensity
- `main_sdfnonsdf / 2x_sdfnonsdf / 3x_sdfnonsdf` use a second shared hue family with ordered intensity

Required annotation logic:

- label the best value in each numeric panel
- explicitly mark the baseline reference
- in the symmetry panel, label the baseline as `reference baseline`

The symmetry panel must not visually imply that the baseline and D4-trained models are semantically identical training setups.

## 8. Data-Extraction Rules

The implementation that follows this design must prefer current repo-backed result files over recomputation.

Primary source expectations:

- numeric metrics from `out/7367/test2_resolution_sweep_metrics.csv` when the requested fields are present and consistent
- symmetry metrics from the existing D4-symmetry evaluation surface already implied by `tem/eta_bin_breakdown/plot_symmetry_comparison.py`

If a required metric is not directly available in a current result surface, the implementation step must state exactly which file or script will derive it and must keep that derivation consistent with the current repo contract.

This design does not approve inventing an unverified metric path.

## 9. Acceptance Criteria

The comparison package is successful only if a reader can answer all of the following directly from the figures plus table:

1. Which object is best on positive-only numeric performance.
2. Whether moving from `main` to `2x` to `3x` improves performance consistently.
3. Whether adding `sdf+nonsdf` helps or hurts numeric quality.
4. How the D4-trained family compares on symmetry MSE.
5. How each model changes relative to `baseline_256_hgradient`.

If the package requires explaining hidden conventions, cross-reading scratch scripts, or interpreting negative-branch diagnostics to understand the main conclusion, it fails the design.

## 10. Out Of Scope

The following are explicitly out of scope for this design:

- negative-branch primary comparison
- mixed positive/negative main panels
- symmetry panels driven by `mean violation` or `max violation`
- a single overloaded all-in-one figure
- redefining the comparison object set without new confirmed evidence

## 11. Next Step After User Review

After the user reviews this spec, the next process step is to write the implementation plan that maps:

- source files
- extraction logic
- plotting outputs
- validation checks

to the approved comparison package above.
