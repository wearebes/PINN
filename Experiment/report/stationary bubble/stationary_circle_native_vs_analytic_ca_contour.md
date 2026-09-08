# Stationary bubble circle: native vs analytic contour

Scope: original/native CLSVOF-LS data and analytic contour-curvature data only. `NN27_RAW_RADIAL` is intentionally excluded.

| method | N | status | Ca kind | Ca | Ca_max kind | Ca_max | tau | rows | note |
|---|---:|---|---|---:|---|---:|---:|---:|---|
| native | 64 | complete | Ca_final | 2.5004e-05 | Ca_max | 7.7596e-05 | 0.999987 | 71133 | complete final trace |
| analytic_contour_curvature | 64 | complete | Ca_final | 2.6202e-05 | Ca_max | 7.6403e-05 | 0.999987 | 71133 | analytic radial curvature trace |
| native | 128 | complete | Ca_final | 1.0519e-05 | Ca_max | 8.5893e-05 | 0.999999 | 201195 | complete final trace |
| analytic_contour_curvature | 128 | complete | Ca_final | 1.0141e-05 | Ca_max | 8.5957e-05 | 0.999999 | 201195 | analytic radial curvature trace |
| native | 256 | partial_trace_no_summary | Ca_current_partial | 6.3136e-06 | Ca_max_so_far_partial | 4.8768e-05 | 0.201609 | 114729 | partial native trace; not final evidence |
| analytic_contour | 256 | contour_only | not_run | contour only | not_run | contour only |  | 720 | N256 analytic contour only; no analytic Ca trace is present on disk |

N256 contour overlay max radial offset: `4.408971e-06`.

Important boundary: N256 native is a partial archived trace, so its plotted Ca is `Ca_current_partial` and its max is `Ca_max_so_far_partial`, not report-final `Ca_final/Ca_max`.
N256 analytic is contour-only; no saved analytic L8 Ca time trace is present in the current workspace.

Data CSV: `stationary_circle_native_vs_analytic_ca_contour.csv`
Contour CSV: `stationary_circle_N256_numeric_vs_analytic_contour.csv`
Figure PNG: `stationary_circle_native_vs_analytic_ca_contour.png`
