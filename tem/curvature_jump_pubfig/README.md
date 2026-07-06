# Curvature-jump diagnostic — publication figure (redesign)

Standalone re-rendering of the stationary curvature-jump diagnostic figure.
The cleanroom pipeline (`cfd_applications_cleanroom/`) is untouched; this
script only consumes its audited artifacts and re-draws the plate.

## Data provenance

The original run referenced in earlier reports
(`stationary_curvature_20260703T165032Z`) was deleted from this machine by the
2026-07-04/05 artifact-cleanup passes (raw bundle, figures, CSVs all gone).
The evidence chain was regenerated fresh on 2026-07-05:

```bash
# 1) regenerate L6 curvature field data (NN27_RAW + NN27_D4)  -> PASS
PYTHONPATH=. conda run -n pinn python -m cfd_applications_cleanroom.cfd_apps.cli \
  reproduce --benchmark stationary --tier curvature-diagnostic \
  --methods NN27_RAW NN27_D4 --levels 6

# 2) rebuild the audited curvature-jump package                -> PASS
PYTHONPATH=. conda run -n pinn python -m cfd_applications_cleanroom.cfd_apps.cli \
  figures --artifact curvature-jump \
  --run-id stationary_curvature_20260705T132529Z \
  --methods NN27_RAW NN27_D4 --level 6

# 3) render this publication figure from the audited artifacts
conda run -n pinn python tem/curvature_jump_pubfig/render_curvature_jump_pubfig.py \
  --run-id stationary_curvature_20260705T132529Z            # main (CCDF panel A)
conda run -n pinn python tem/curvature_jump_pubfig/render_curvature_jump_pubfig.py \
  --run-id stationary_curvature_20260705T132529Z --panel-a ecdf  # ECDF variant
```

The regenerated tail statistics reproduce the deleted 20260703T165032Z run to
displayed precision (deterministic diagnostic): native 1.87/2.27/2.53e-3,
NN 1.85/2.00/2.04e-3, NND4 1.79/2.02/2.18e-3 (p95/p99/max).

## Design vs the previous plate

- Panel A: survival function (CCDF, log-y) instead of ECDF — the scientific
  claim lives in the upper tail; p95/p99 are readable directly as crossings of
  the 0.05 / 0.01 guides. An ECDF variant (`*_ecdfA.*`) is provided.
- Panel B: the typeset number table is replaced by a slopegraph of
  p95 -> p99 -> max; only the max values are labelled in-figure. Exact values
  remain in the audited summary JSON and the plotted-data CSV.
- Panel C: the 0-360 symmetry expansion is NOT drawn. Only the measured
  0-90 deg quadrant is shown: all 158 arc-neighbour correction jumps per
  method as scatter, 15-deg-binned p95 lines, and the native-field arc-jump
  p95 as a dashed reference. (The pipeline plate used 30-deg bins on the
  expanded domain; re-binning here is done directly from the audited
  source-data rows and is recorded in the plotted-data CSV.)
- No in-figure provenance footnotes; that information belongs in the caption.
- Okabe-Ito colourblind-safe palette; 183 mm double-column width; 600 dpi PNG
  plus vector PDF/SVG (SVG keeps text as text).

## Built-in checks (script fails hard on violation)

- Audited summary must be overall_status=PASS before anything is drawn.
- Recomputed p95/p99/max for all three panel-A series must match the audited
  summary JSON exactly (9/9 values, rel_tol 1e-9), incl. pair counts.
- Native grid-edge jumps must be identical between the NN27_RAW and NN27_D4
  run directories (single-snapshot consistency).
- Measured theta must lie within 0-90 deg.
- Outputs: `*_plotted_data.csv` (every drawn point) and `*_provenance.json`
  (input paths + SHA256, series counts, cross-check results).

## Draft caption (adapt before submission)

> Cell-to-cell roughness diagnostic of the NN curvature correction on the
> stationary bubble (same-host CLSVOF-LS, Basilisk, level 6, force band
> |d| <= 2*Delta; controlled diagnostic, not a production Ca/velocity
> benchmark — VOF-HF is not the baseline here).
> (A) Survival distribution of jump magnitudes across 4-neighbour grid edges:
> native h*kappa field increments (grey) vs NN / NND4 correction increments
> delta(h*kappa) (blue / vermillion); dotted guides mark the p95 and p99
> levels. (B) Tail statistics of the same distributions; native tails grow
> steeply toward the maximum (2.53e-3) while correction tails saturate
> (2.04e-3 / 2.18e-3). (C) Arc-neighbour correction jumps against interface
> angle in the measured 0-90 deg quadrant (points: individual cell pairs;
> lines: p95 in 15-deg bins; dashed grey: native-field reference). NND4
> averages the network over the 8 D4 symmetry transforms. The correction
> nowhere exceeds the roughness the native curvature field already carries.

## Files

- `render_curvature_jump_pubfig.py` — renderer (no pipeline edits)
- `out/stationary_curvature_20260705T132529Z_curvature_jump_pubfig.{png,pdf,svg}` — main
- `out/..._pubfig_ecdfA.{png,pdf,svg}` — ECDF panel-A variant
- `out/..._plotted_data.csv`, `out/..._provenance.json` — per variant
