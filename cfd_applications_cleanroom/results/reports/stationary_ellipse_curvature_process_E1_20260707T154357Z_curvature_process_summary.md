# Stationary Curvature Process Diagnostic

- run_id: `stationary_ellipse_curvature_process_E1_20260707T154357Z`
- evidence_level: `controlled_diagnostic`
- primary_metric: `Ca=mu*Umax/sigma`
- report_gate_metric: `Ca_tail_max`
- figure_png: `/Users/jcy/research/PINN/cfd_applications_cleanroom/results/figures/stationary_ellipse_curvature_process_E1_20260707T154357Z_stationary_ellipse_curvature_process_e1_plate.png`
- source_data: `/Users/jcy/research/PINN/cfd_applications_cleanroom/results/source_data/stationary_ellipse_curvature_process_E1_20260707T154357Z_stationary_ellipse_curvature_process_e1_plate_source_data.csv`

| method | Ca max | Ca tail max | reached final time | sigma |
|---|---:|---:|---|---:|
| NN_DISABLE | 2.401225e-03 | 1.052019e-05 | True | 1 |
| NN27_RAW | 2.401306e-03 | 1.388201e-05 | True | 1 |

## Snapshot Curvature Error

| method | fraction | tau | max |Δ(hκ)| | stdΓ κ_NN |
|---|---:|---:|---:|---:|
| NN_DISABLE | 0 | 0 | 1.450774e-03 | 5.924352e-01 |
| NN_DISABLE | 0.333333 | 0.333335 | 8.980194e-04 | 9.288691e-03 |
| NN_DISABLE | 0.666667 | 0.666669 | 8.980143e-04 | 9.289607e-03 |
| NN_DISABLE | 1 | 1 | 8.980143e-04 | 9.289607e-03 |
| NN27_RAW | 0 | 0 | 1.450774e-03 | 5.924352e-01 |
| NN27_RAW | 0.333333 | 0.333335 | 8.941623e-04 | 9.331606e-03 |
| NN27_RAW | 0.666667 | 0.666669 | 8.941587e-04 | 9.332923e-03 |
| NN27_RAW | 1 | 1 | 8.941587e-04 | 9.332923e-03 |
