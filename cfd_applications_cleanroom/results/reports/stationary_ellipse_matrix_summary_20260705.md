# Stationary Ellipse Matrix Summary (2026-07-05)

## Deliverables
- CSV: `cfd_applications_cleanroom/results/source_data/stationary_ellipse_matrix_summary_20260705.csv`
- Diagnostic PNG: `cfd_applications_cleanroom/results/figures/stationary_ellipse_curvature_diagnostic_20260705.png`
- Process status PNG: `cfd_applications_cleanroom/results/figures/stationary_ellipse_process_status_20260705.png`

## Evidence Status
- `curvature-diagnostic` / `manifest_backed_complete`: 12 rows
- `curvature-process` / `missing`: 9 rows
- `curvature-process` / `partial_trace_no_summary`: 3 rows
- `curvature-process` / `raw_summary_complete_unmanifested`: 12 rows

## Interpretation
- Curvature diagnostic is complete and manifest-backed for E1/E2 at 64, 128, and 256 using NN and NND4.
- Curvature process has completed raw summaries only for E1/E2 at 64 and 128 using native, probe-only, and NN. These rows are not manifest-backed because the full process run was interrupted before all methods/resolutions finished.
- Process 256 native rows are partial traces only, and NND4 process is not complete; E1/64/NND4 has only an interrupted trace from the earlier attempt.

## Key Diagnostic Ratios
### E1
- 64: NND4/NN std(delta kappa)=0.982108, max_abs(delta h*kappa)=1.02762
- 128: NND4/NN std(delta kappa)=0.968815, max_abs(delta h*kappa)=0.971113
- 256: NND4/NN std(delta kappa)=0.885949, max_abs(delta h*kappa)=0.890311
### E2
- 64: NND4/NN std(delta kappa)=1.00661, max_abs(delta h*kappa)=1.05244
- 128: NND4/NN std(delta kappa)=0.962434, max_abs(delta h*kappa)=0.956069
- 256: NND4/NN std(delta kappa)=0.907737, max_abs(delta h*kappa)=0.97039
