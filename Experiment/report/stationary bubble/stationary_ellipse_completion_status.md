# Stationary Ellipse Completion Status

- updated_utc: `2026-07-06T10:28:41Z`
- summary_csv: `Experiment/report/stationary bubble/summary.csv`
- diagnostic_ready: `12/12`
- process_ready: `16/24`
- l8_process_ready: `0/8`
- process_png_plates_ready: `0/6`
- final_delivery_status: `not_ready`
- finalizer_dir: `cfd_applications_cleanroom/results/background/stationary_ellipse_finalizer_20260705T183521Z`

## Current Missing Rows

| case | grid | method | status | tau/progress | trace rows |
|---|---:|---|---|---:|---:|
| E1 | 256 | NN_DISABLE | partial_trace_no_summary | 0.26210620588191313 (26.210620588191315%) | 149156 |
| E1 | 256 | NN_PROBE_ONLY | partial_trace_no_summary | 0.26275815454190737 (26.275815454190738%) | 149527 |
| E1 | 256 | NN27_RAW | partial_trace_no_summary | 0.24871050605674816 (24.871050605674817%) | 141533 |
| E1 | 256 | NN27_D4 | partial_trace_no_summary | 0.05872106852432063 (5.872106852432063%) | 33417 |
| E2 | 256 | NN_DISABLE | partial_trace_no_summary | 0.2530598112995166 (25.30598112995166%) | 144008 |
| E2 | 256 | NN_PROBE_ONLY | partial_trace_no_summary | 0.25315470411590796 (25.315470411590795%) | 144062 |
| E2 | 256 | NN27_RAW | partial_trace_no_summary | 0.2417605370514388 (24.17605370514388%) | 137578 |
| E2 | 256 | NN27_D4 | partial_trace_no_summary | 0.024181852710581611 (2.418185271058161%) | 13762 |

## Required Completion Step

Run on Windows/WSL from the repository root:

```bash
MAX_JOBS=4 bash cfd_applications_cleanroom/scripts/run_and_pack_stationary_ellipse_l8_wsl.sh
```

After copying the generated package back to this workspace, run:

```bash
bash cfd_applications_cleanroom/scripts/import_stationary_ellipse_l8_package.sh \
  cfd_applications_cleanroom/results/transfer/stationary_ellipse_l8_results_*.tar.gz
```

Final acceptance command:

```bash
python cfd_applications_cleanroom/scripts/verify_stationary_ellipse_delivery.py
```

Final acceptance requires `process_ready=24/24`, `stationary_ellipse_l8_complete=8/8`, and six generated process PNG plates.
