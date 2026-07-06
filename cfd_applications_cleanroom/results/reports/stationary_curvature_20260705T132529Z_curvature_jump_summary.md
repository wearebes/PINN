# Stationary Curvature Jump Diagnostic

- run_id: `stationary_curvature_20260705T132529Z`
- evidence_level: `controlled_diagnostic`
- overall_status: `PASS`
- summary_json: `cfd_applications_cleanroom/results/reports/stationary_curvature_20260705T132529Z_curvature_jump_summary.json`
- source_data_csv: `cfd_applications_cleanroom/results/source_data/stationary_curvature_20260705T132529Z_curvature_jump_source_data.csv`
- plotted_source_data_csv: `cfd_applications_cleanroom/results/source_data/stationary_curvature_20260705T132529Z_curvature_jump_plotted_source_data.csv`
- source_data_manifest: `cfd_applications_cleanroom/results/source_data/stationary_curvature_20260705T132529Z_curvature_jump_manifest.json`
- figure_png: `cfd_applications_cleanroom/results/figures/stationary_curvature_20260705T132529Z_curvature_jump_plate.png`

This is controlled_diagnostic evidence for local curvature-field roughness.
It does not replace same-host stationary-bubble Ca evidence.
VOF-HF remains a positive-control reference, not the research target replacement.
Angle-domain note: the source CSV is measured on the quadrant host (0-90 degrees). The figure uses quadrant symmetry expansion to display 0-360 degrees; it is not an independent full-domain export.
D4 note: `NND4` is the baseline_hgradient D4 consensus, i.e. the average of 8 transformed NN predictions.
Publication figure note: the visible plate is a three-panel tail-evidence figure: grid-neighbor ECDF, p95/p99/max table, and binned angular p95 profile.

## Gate Table

| gate | status |
|---|---|
| JG-D1 | PASS |
| JG-D2 | PASS |
| JG-D3 | PASS |
| JG-D4 | PASS |
| JG-D5 | PASS |
| JG-D6 | PASS |
| JG-D7 | PASS |
| JG-D8 | PASS |
| JG-D9 | PASS |
| JG-D10 | PASS |
| JG-D11 | PASS |
| JG-D12 | PASS |
| JG-D13 | PASS |
| JG-D14 | PASS |
| JG-N1 | PASS |
| JG-N2 | PASS |
| JG-N3 | PASS |
| JG-N4 | PASS |
| JG-N5 | PASS |
| JG-N6 | PASS |
| JG-N7 | PASS |
| JG-N8 | PASS |
| JG-N9 | PASS |
| JG-F1 | PASS |
| JG-F2 | PASS |
| JG-F3 | PASS |
| JG-F4 | PASS |
| JG-F5 | PASS |
| JG-R1 | PASS |
| JG-R2 | PASS |
| JG-R3 | PASS |
| JG-R4 | PASS |
| JG-R5 | PASS |
| JG-R6 | PASS |
| JG-R7 | PASS |
| JG-R8 | PASS |
| JG-R9 | PASS |
| JG-R10 | PASS |
| JG-R11 | PASS |
| JG-R12 | PASS |

## Jump Summary

| method | band | neighbor | signal | pairs | mean | rms | p50 | p95 | p99 | max | total variation | ratio to native mean | ratio to native p95 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NN | force_band | grid_edge | hk_native | 262 | 9.859131e-04 | 1.164129e-03 | 1.032239e-03 | 1.866208e-03 | 2.274063e-03 | 2.533572e-03 | 2.583092e-01 | 1.000000e+00 | 1.000000e+00 |
| NN | force_band | grid_edge | hk_nn | 262 | 2.571108e-04 | 3.467299e-04 | 1.692533e-04 | 7.087886e-04 | 9.283529e-04 | 9.650286e-04 | 6.736304e-02 | 2.607845e-01 | 3.798016e-01 |
| NN | force_band | grid_edge | delta_hk | 262 | 8.618534e-04 | 1.035078e-03 | 7.916936e-04 | 1.846719e-03 | 1.997887e-03 | 2.036951e-03 | 2.258056e-01 | 8.741677e-01 | 9.895572e-01 |
| NN | force_band | theta_order | hk_native | 158 | 2.531231e-03 | 2.808961e-03 | 2.034923e-03 | 4.894347e-03 | 5.630294e-03 | 5.690621e-03 | 3.999346e-01 | 1.000000e+00 | 1.000000e+00 |
| NN | force_band | theta_order | hk_nn | 158 | 3.874119e-04 | 4.966193e-04 | 3.012560e-04 | 9.407566e-04 | 1.073843e-03 | 1.194669e-03 | 6.121108e-02 | 1.530527e-01 | 1.922129e-01 |
| NN | force_band | theta_order | delta_hk | 158 | 2.230958e-03 | 2.497158e-03 | 1.997378e-03 | 4.083234e-03 | 4.735745e-03 | 4.775646e-03 | 3.524914e-01 | 8.813726e-01 | 8.342755e-01 |
| NN | interface_band | grid_edge | hk_native | 106 | 9.592267e-04 | 1.177985e-03 | 8.500135e-04 | 1.920932e-03 | 2.510280e-03 | 2.533572e-03 | 1.016780e-01 | 1.000000e+00 | 1.000000e+00 |
| NN | interface_band | grid_edge | hk_nn | 106 | 1.254010e-04 | 1.748276e-04 | 9.308596e-05 | 4.068257e-04 | 4.999652e-04 | 5.451838e-04 | 1.329251e-02 | 1.307314e-01 | 2.117855e-01 |
| NN | interface_band | grid_edge | delta_hk | 106 | 9.311893e-04 | 1.116874e-03 | 8.438881e-04 | 1.954315e-03 | 2.020578e-03 | 2.036951e-03 | 9.870607e-02 | 9.707709e-01 | 1.017378e+00 |
| NN | interface_band | theta_order | hk_native | 79 | 1.505607e-03 | 1.646472e-03 | 1.605828e-03 | 2.649250e-03 | 2.736288e-03 | 2.736288e-03 | 1.189429e-01 | 1.000000e+00 | 1.000000e+00 |
| NN | interface_band | theta_order | hk_nn | 79 | 1.155790e-04 | 1.547495e-04 | 9.360397e-05 | 3.417886e-04 | 4.094740e-04 | 4.185226e-04 | 9.130741e-03 | 7.676574e-02 | 1.290133e-01 |
| NN | interface_band | theta_order | delta_hk | 79 | 1.472039e-03 | 1.595324e-03 | 1.586136e-03 | 2.503813e-03 | 2.550648e-03 | 2.577707e-03 | 1.162911e-01 | 9.777049e-01 | 9.451023e-01 |
| NND4 | force_band | grid_edge | hk_native | 262 | 9.859131e-04 | 1.164129e-03 | 1.032239e-03 | 1.866208e-03 | 2.274063e-03 | 2.533572e-03 | 2.583092e-01 | 1.000000e+00 | 1.000000e+00 |
| NND4 | force_band | grid_edge | hk_nn | 262 | 2.573003e-04 | 3.511958e-04 | 1.641863e-04 | 8.481201e-04 | 9.249207e-04 | 9.774130e-04 | 6.741269e-02 | 2.609767e-01 | 4.544618e-01 |
| NND4 | force_band | grid_edge | delta_hk | 262 | 8.369520e-04 | 1.012047e-03 | 7.877355e-04 | 1.788432e-03 | 2.024918e-03 | 2.179855e-03 | 2.192814e-01 | 8.489105e-01 | 9.583245e-01 |
| NND4 | force_band | theta_order | hk_native | 158 | 2.531231e-03 | 2.808961e-03 | 2.034923e-03 | 4.894347e-03 | 5.630294e-03 | 5.690621e-03 | 3.999346e-01 | 1.000000e+00 | 1.000000e+00 |
| NND4 | force_band | theta_order | hk_nn | 158 | 4.047382e-04 | 5.262735e-04 | 3.117436e-04 | 9.937703e-04 | 1.177445e-03 | 1.326798e-03 | 6.394863e-02 | 1.598977e-01 | 2.030445e-01 |
| NND4 | force_band | theta_order | delta_hk | 158 | 2.183465e-03 | 2.441623e-03 | 1.958058e-03 | 3.948709e-03 | 4.589654e-03 | 4.643100e-03 | 3.449874e-01 | 8.626096e-01 | 8.067897e-01 |
| NND4 | interface_band | grid_edge | hk_native | 106 | 9.592267e-04 | 1.177985e-03 | 8.500135e-04 | 1.920932e-03 | 2.510280e-03 | 2.533572e-03 | 1.016780e-01 | 1.000000e+00 | 1.000000e+00 |
| NND4 | interface_band | grid_edge | hk_nn | 106 | 1.135400e-04 | 1.550059e-04 | 8.419121e-05 | 3.406273e-04 | 4.735360e-04 | 4.773512e-04 | 1.203524e-02 | 1.183662e-01 | 1.773240e-01 |
| NND4 | interface_band | grid_edge | delta_hk | 106 | 9.417575e-04 | 1.131038e-03 | 8.447813e-04 | 1.958821e-03 | 2.172109e-03 | 2.179855e-03 | 9.982629e-02 | 9.817883e-01 | 1.019724e+00 |
| NND4 | interface_band | theta_order | hk_native | 79 | 1.505607e-03 | 1.646472e-03 | 1.605828e-03 | 2.649250e-03 | 2.736288e-03 | 2.736288e-03 | 1.189429e-01 | 1.000000e+00 | 1.000000e+00 |
| NND4 | interface_band | theta_order | hk_nn | 79 | 1.017079e-04 | 1.357081e-04 | 8.544267e-05 | 2.738012e-04 | 4.010482e-04 | 4.010482e-04 | 8.034921e-03 | 6.755274e-02 | 1.033504e-01 |
| NND4 | interface_band | theta_order | delta_hk | 79 | 1.487919e-03 | 1.612491e-03 | 1.607481e-03 | 2.506207e-03 | 2.630585e-03 | 2.630585e-03 | 1.175456e-01 | 9.882522e-01 | 9.460062e-01 |

## Interpretation

This table compares the selected NN curvature closures against the same CLSVOF-LS native curvature field. Larger `delta_hk` jump ratios indicate stronger local roughness in the NN correction. This is a controlled diagnostic only; it must be read beside same-host CLSVOF-LS Ca evidence.
