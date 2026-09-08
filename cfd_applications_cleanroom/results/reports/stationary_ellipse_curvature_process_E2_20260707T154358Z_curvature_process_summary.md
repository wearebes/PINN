# Stationary Curvature Process Diagnostic

- run_id: `stationary_ellipse_curvature_process_E2_20260707T154358Z`
- evidence_level: `controlled_diagnostic`
- primary_metric: `Ca=mu*Umax/sigma`
- report_gate_metric: `Ca_tail_max`
- figure_png: `/Users/jcy/research/PINN/cfd_applications_cleanroom/results/figures/stationary_ellipse_curvature_process_E2_20260707T154358Z_stationary_ellipse_curvature_process_e2_plate.png`
- source_data: `/Users/jcy/research/PINN/cfd_applications_cleanroom/results/source_data/stationary_ellipse_curvature_process_E2_20260707T154358Z_stationary_ellipse_curvature_process_e2_plate_source_data.csv`

| method | Ca max | Ca tail max | reached final time | sigma |
|---|---:|---:|---|---:|
| NN_DISABLE | 4.610650e-03 | 1.052054e-05 | True | 1 |
| NN27_RAW | 4.610492e-03 | 1.388305e-05 | True | 1 |

## Snapshot Curvature Error

| method | fraction | tau | max |Δ(hκ)| | stdΓ κ_NN |
|---|---:|---:|---:|---:|
| NN_DISABLE | 0 | 0 | 2.224227e-03 | 1.062979e+00 |
| NN_DISABLE | 0.333333 | 0.333335 | 8.980239e-04 | 9.288153e-03 |
| NN_DISABLE | 0.666667 | 0.666669 | 8.980151e-04 | 9.289627e-03 |
| NN_DISABLE | 1 | 1 | 8.980151e-04 | 9.289627e-03 |
| NN27_RAW | 0 | 0 | 2.224227e-03 | 1.062979e+00 |
| NN27_RAW | 0.333333 | 0.333335 | 8.941630e-04 | 9.331063e-03 |
| NN27_RAW | 0.666667 | 0.666669 | 8.941578e-04 | 9.332948e-03 |
| NN27_RAW | 1 | 1 | 8.941578e-04 | 9.332949e-03 |
