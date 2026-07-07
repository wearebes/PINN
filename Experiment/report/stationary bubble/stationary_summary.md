# Stationary Bubble Data Index

## Storage Files

| item | file | size | sha256 / note |
|---|---|---:|---|
| circle source | `stationary_bubble_source.csv.gz` | 92.8 MB | `e847446ce3fbe31a8ef27d5ae97e1347e52c1f0fd738f70a864c9f635d83d0c9` |
| ellipse source | `stationary_ellipse_source.csv.gz` | 121.3 MB | `a02c5fe621bffca58e7382cb17972e1c122e058096b858fc9842326893c6e103` |

Read data:

```python
import pandas as pd
df = pd.read_csv("stationary_bubble_source.csv.gz")
df = pd.read_csv("stationary_ellipse_source.csv.gz")
```

## Source CSV Keys

| column | use |
|---|---|
| `row_type` | `index`, `trace`, `curvature`, `diagnostic` |
| `case_id` | `circle`, `E1`, `E2` |
| `grid_n` | grid resolution |
| `method` | solver/method label |
| `series` | repeat or main series |
| `status` | completion status |
| `report_ready` | final-report eligibility |

Rule: only `report_ready=true` rows are final-report evidence.

## Provenance

| item | value |
|---|---|
| circle archive built_utc | 2026-07-07T12:03:34Z |
| circle NN process run | `stationary_curvature_process_20260705T043136Z` |
| circle native canary run | `stationary_canary_20260705T214031Z` |
| circle retired summary state | interim report; N256 native is `partial_trace_no_summary` |
| ellipse archive built_utc | 2026-07-07T12:04:16Z |
| ellipse process tier | `stationary_ellipse_curvature_process_*` raw runs listed in archive `index` rows |
| ellipse missing expected units | `E1/L8/NN_PROBE_ONLY`, `E1/L8/NN27_RAW`, `E1/L8/NN27_D4`, `E2/L8/NN_PROBE_ONLY`, `E2/L8/NN27_RAW`, `E2/L8/NN27_D4` |
| retired wide CSV | `summary.csv`, `summary_index.csv`, and `summary_manifest.json` are superseded by the two source archives above |

## Circle Index Rows

| case | N | method | series | status | ready | evidence | Ca_final | Ca_max | tau | rows |
|---|---:|---|---|---|---|---|---:|---:|---:|---:|
| circle | 64 | NN27_RAW | main | complete | true | controlled_diagnostic | 3.329e-05 | 7.759e-05 | 9.9999e-01 | 71133 |
| circle | 64 | CLSVOF_LS_NATIVE | r0/r1/r2 | complete | true | runtime_smoke | 2.500e-05 | 7.760e-05 | 9.9999e-01 | 71133 |
| circle | 128 | NN27_RAW | main | complete | true | controlled_diagnostic | 1.388e-05 | 8.591e-05 | 1.0000e+00 | 201195 |
| circle | 128 | CLSVOF_LS_NATIVE | r0/r1/r2 | complete | true | runtime_smoke | 1.052e-05 | 8.589e-05 | 1.0000e+00 | 201195 |
| circle | 256 | NN27_RAW | main | complete | true | controlled_diagnostic | 8.193e-06 | 4.879e-05 | 1.0000e+00 | 569064 |
| circle | 256 | CLSVOF_LS_NATIVE | r0 | partial_trace_no_summary | false | runtime_smoke | NA | NA | 2.0161e-01 | 114729 |

## Ellipse Process Status

| case | N | method | status |
|---|---:|---|---|
| E1 | 64 | NN_DISABLE | complete |
| E1 | 64 | NN_PROBE_ONLY | complete |
| E1 | 64 | NN27_RAW | complete |
| E1 | 64 | NN27_D4 | partial_trace_no_summary |
| E1 | 128 | NN_DISABLE | complete |
| E1 | 128 | NN_PROBE_ONLY | complete |
| E1 | 128 | NN27_RAW | complete |
| E1 | 128 | NN27_D4 | complete |
| E1 | 256 | all process methods | partial or missing locally |
| E2 | 64 | NN_DISABLE | complete |
| E2 | 64 | NN_PROBE_ONLY | complete |
| E2 | 64 | NN27_RAW | complete |
| E2 | 64 | NN27_D4 | complete |
| E2 | 128 | NN_DISABLE | complete |
| E2 | 128 | NN_PROBE_ONLY | complete |
| E2 | 128 | NN27_RAW | complete |
| E2 | 128 | NN27_D4 | complete |
| E2 | 256 | all process methods | partial or missing locally |

## Ellipse Completion Counts

| item | count/status |
|---|---:|
| diagnostic_ready | 12/12 |
| process_ready | 16/24 |
| l8_process_ready | 0/8 |
| process_png_plates_ready | 0/6 |
| final_delivery_status | not_ready |

Known local gap: `E1/L6/NN27_D4` process-tier complete evidence is missing; only
a 3,371-row partial trace is archived. Its diagnostic-tier evidence is complete.

## Recovery / Completion Commands

```bash
MAX_JOBS=4 bash cfd_applications_cleanroom/scripts/run_and_pack_stationary_ellipse_l8_wsl.sh

bash cfd_applications_cleanroom/scripts/import_stationary_ellipse_l8_package.sh \
  cfd_applications_cleanroom/results/transfer/stationary_ellipse_l8_results_*.tar.gz

python cfd_applications_cleanroom/scripts/verify_stationary_ellipse_delivery.py
```

Acceptance target: `process_ready=24/24`,
`stationary_ellipse_l8_complete=8/8`, six process PNG plates.

## Raw Deletion Gate

| gate | command / rule |
|---|---|
| backup | `python3 tem/report_source_archive/verify_backup.py <backup_dir>` |
| circle redraw | `PYTHONPATH=. /opt/anaconda3/bin/conda run -n base --no-capture-output python tem/report_source_archive/gate_stationary_redraw.py --family circle` |
| ellipse redraw | `PYTHONPATH=. /opt/anaconda3/bin/conda run -n base --no-capture-output python tem/report_source_archive/gate_stationary_redraw.py --family ellipse` |
| scope | delete only raw dirs represented by archive `index` rows |
| ellipse | no raw deletion before WSL L8 import + archive rebuild |
