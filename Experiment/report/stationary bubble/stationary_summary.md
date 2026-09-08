# Stationary Bubble & Ellipse — Source Archives

One combined summary for both benchmark families. Each section below is fully self-contained (own CSV archive, own provenance sidecar, own deletion gate) — only this MD file is shared.

- [Stationary Bubble (circle)](#stationary-bubble-circle)
- [Stationary Ellipse (E1/E2)](#stationary-ellipse-e1e2)

---

# Stationary Bubble (circle) — source archive

**This CSV, this MD, and `stationary_bubble_provenance.json` are the sole long-term
record for the circle report plates. Raw deletion is scoped and gated —
see "Deletion preconditions" below; never delete the raw root wholesale.**

- archive: `stationary_bubble_source.csv.gz` — 92.8 MB gz, sha256 `e847446ce3fbe31a8ef27d5ae97e1347e52c1f0fd738f70a864c9f635d83d0c9`
- built: 2026-07-07T12:03:34Z by `tem/report_source_archive/build_stationary_source.py --family circle`
- supersedes the retired `summary.csv` / `summary_index.csv` /
  `summary_manifest.json` / `summary_schema.md` (1.27 GB wide-format CSV).
- the csv.gz is gitignored (size); this MD records its sha256, and
  `verify_backup.py` checks external copies against it.

Load with: `pandas.read_csv("stationary_bubble_source.csv.gz")` (pandas decompresses transparently).
Filter with `row_type`, then `case_id/grid_n/method/series`.

## CSV schema

One gzipped CSV, one header, four `row_type` values. Column groups (unused
groups are empty on a given row; gzip absorbs the cost):

| group | columns | used by |
|---|---|---|
| keys (all rows) | `row_type, case_id, grid_n, method, series, status, report_ready` | filtering |
| trace | `i, t, dt, tau, Umax, Ca, mass, kappa_linf, kappa_probe_linf, dc` | `row_type=trace` |
| curvature | `snapshot_index, snapshot_fraction, x, y, theta_deg, d, abs_d_over_delta, hk_native, hk_nn, delta_hk, native_kappa, kappa_nn, delta_kappa, kappa_nn_from_hk_over_delta, scale_identity_error, sign_product, hk_nn_raw, hk_nn_force, delta_hk_raw, delta_hk_force, relax_lambda, relax_neighbor_count` (+ shares `i, t, tau, Ca`) | `row_type=curvature` |
| diagnostic | curvature group's shared geometry columns + `theta, hk_analytic, kappa_analytic, delta_hk_native_analytic, delta_hk_nn_analytic, delta_kappa_native_analytic, delta_kappa_nn_analytic` | `row_type=diagnostic` (ellipse only) |
| index | `run_id, source_trace_csv, source_curvature_csv, benchmark, deployable, evidence_level, repeat_id, sigma, mu, expected_tmax, reached_final_time, final_t, final_tau, Ca_final, Ca_max, Ca_tail_mean, Ca_tail_max, mass_final, kappa_linf_tail_max, kappa_linf_tail_mean, kappa_probe_linf_tail_max, current_t, current_tau, current_Ca, trace_row_count, curvature_row_count, resolved_config_sha256, binary_sha256, official_source_manifest_sha256, geometry, report_png, note` | `row_type=index` |

### Row types

- `index` — one row per (case_id, grid_n, method, series) unit: provenance,
  completion status, evidence level, final metrics. All long strings live
  here only.
- `trace` — per-timestep solver trace, verbatim from the raw
  `surface_tension_trace.csv` / `stationary_trace.csv` (no float round-trip).
- `curvature` — per-interface-point curvature-process rows, verbatim from the
  raw `curvature_process.csv`. The FULL radial band is kept: rows with
  `abs_d_over_delta > 1` are NOT filtered and the 4x quadrant mirroring to
  0-360 deg is NOT applied — both are redraw-time derivations
  (`write_curvature_process_source_data` in
  `cfd_applications_cleanroom/cfd_apps/curvature_diagnostic_shared.py`).
- `diagnostic` — (ellipse) static curvature-diagnostic tier with analytic
  ellipse reference columns, verbatim from `curvature_field.csv`.

### Columns dropped from the raw traces, and their exact recovery

| dropped | recovery |
|---|---|
| `primary_metric_name` | constant `Ca` (asserted at build) |
| `primary_metric_value` | bitwise identical to `Ca` on every row (asserted at build) |
| `sigma, mu, benchmark, deployable, evidence_level, repeat_id` | constant per file (asserted at build); archived verbatim on the unit's index row |
| `run_id, case_id, method, level, grid_n` | constant per file; index-row keys. `level = log2(grid_n)` |

`Umax` and `tau` are kept verbatim (Umax = Ca*sigma/mu and tau = t/TMAX only
up to 1 ulp, so they are archived rather than derived).

### Semantics

- `status`: `complete` (reached final time, summary verified),
  `partial_trace_no_summary` (run did not finish; only current values),
  `complete_manifest` (diagnostic tier).
- `report_ready`: `false` rows must never be presented as final evidence.
- Tail metrics use the last 20% of trace rows (`tail = ca[int(0.8*len):]`),
  matching `stationary.py`. `Ca_final`/`Ca_max`/`Ca_tail_max` recompute
  bitwise from the trace rows; `Ca_tail_mean` matches to <1e-12 relative
  (summation-order noise: the run-time writer used a different sum order).

## Final metrics (from `row_type=index`; regenerate, don't hand-edit)

| case | N | method | series | status | ready | evidence | Ca_final | Ca_max | Ca_tail_mean | current_tau | rows |
|---|---:|---|---|---|---|---|---|---|---|---|---:|
| circle | 64 | NN27_RAW | main | complete | true | controlled_diagnostic | 3.329e-05 | 7.759e-05 | 3.329e-05 | 9.9999e-01 | 71133 |
| circle | 64 | CLSVOF_LS_NATIVE | r0 | complete | true | runtime_smoke | 2.500e-05 | 7.760e-05 | 2.500e-05 | 9.9999e-01 | 71133 |
| circle | 64 | CLSVOF_LS_NATIVE | r1 | complete | true | runtime_smoke | 2.500e-05 | 7.760e-05 | 2.500e-05 | 9.9999e-01 | 71133 |
| circle | 64 | CLSVOF_LS_NATIVE | r2 | complete | true | runtime_smoke | 2.500e-05 | 7.760e-05 | 2.500e-05 | 9.9999e-01 | 71133 |
| circle | 128 | NN27_RAW | main | complete | true | controlled_diagnostic | 1.388e-05 | 8.591e-05 | 1.388e-05 | 1.0000e+00 | 201195 |
| circle | 128 | CLSVOF_LS_NATIVE | r0 | complete | true | runtime_smoke | 1.052e-05 | 8.589e-05 | 1.052e-05 | 1.0000e+00 | 201195 |
| circle | 128 | CLSVOF_LS_NATIVE | r1 | complete | true | runtime_smoke | 1.052e-05 | 8.589e-05 | 1.052e-05 | 1.0000e+00 | 201195 |
| circle | 128 | CLSVOF_LS_NATIVE | r2 | complete | true | runtime_smoke | 1.052e-05 | 8.589e-05 | 1.052e-05 | 1.0000e+00 | 201195 |
| circle | 256 | NN27_RAW | main | complete | true | controlled_diagnostic | 8.193e-06 | 4.879e-05 | 8.192e-06 | 1.0000e+00 | 569064 |
| circle | 256 | CLSVOF_LS_NATIVE | r0 | partial_trace_no_summary | false | runtime_smoke | — | — | — | 2.0161e-01 | 114729 |


## Provenance

- NN27_RAW curvature-process run: `stationary_curvature_process_20260705T043136Z` (pinned in
  `build_stationary_source.py`; the build FAILS if unpinned sibling run dirs
  appear, so the evidence source cannot switch silently)
- CLSVOF-LS native canary run: `stationary_canary_20260705T214031Z` (pinned) —
  repeats r0/r1/r2 archived for L6/L7 (repeat-consistency evidence,
  previously `not_evaluated`); L8 has r0 only, partial (`tau≈0.20` at
  archive time).
- Report plates: `stationarybuubble_64.png` (sic), `stationarybubble_128.png`
  = complete; `stationarybubble_256.png` = interim (CLSVOF-LS L8 partial),
  re-rendered 2026-07-07 from the 114,729-row trace state archived here (the
  13:12 render used the older 92,756-row state).

## Irrecoverable boundary after raw deletion

Archived verbatim: all trace rows, all curvature-process rows (full radial
band), final metrics, evidence levels, config JSONs (below),
run/binary/source sha256s.

NOT archived — gone forever once the two archived run dirs are deleted:
- `build/` compile trees, `stdout.log` / `stderr.log`, `status.json`,
  `source_hashes.json` (top-level digests preserved on index rows)
- field dumps and any per-run artifacts outside the two per-step CSVs

NOT covered by this archive at all — do NOT delete along with it:
- every other dir under `results/raw/stationary/`
  (`stationary_stock_*` ~239M, `stationary_clsvof_baseline_*` ~63M,
  `stationary_curvature_20260705T132529Z`, ...). Those belong to other
  evidence lines and need their own archive-or-delete decision.

## Deletion preconditions (ALL must hold; none are optional)

1. **External backup verified** — the csv.gz files are gitignored (the
   ellipse one exceeds GitHub's 100 MB limit), so an out-of-repo backup must
   exist and verify before any raw deletion:
   `python3 tem/report_source_archive/verify_backup.py <backup_dir>`
   (recomputes sha256 of the backup copies against the local archives and
   the hashes recorded in these MDs).
2. **Redraw gate PASS** (commands below).
3. **Scope audit** — deletion covers ONLY the raw run dirs this archive
   records in its index rows. Other dirs under the same raw root are NOT
   covered and need their own audit (see the irrecoverable-boundary section).

```bash
cd /Users/jcy/research/PINN
# rebuild archive from raw (only while raw still exists)
python3 tem/report_source_archive/build_stationary_source.py --family circle
# regenerate this MD (reads only csv.gz + tracked provenance json)
python3 tem/report_source_archive/gen_summary_md.py
# hard gate: redraw all report plates from the csv.gz ALONE and compare
PYTHONPATH=. /opt/anaconda3/bin/conda run -n base --no-capture-output \
  python tem/report_source_archive/gate_stationary_redraw.py --family circle
```

Gate checks: G1 byte-identical plate redraw from the archive only; G2 index
metrics recomputed from archived trace rows; G3 unit/row-count/snapshot
coverage; G4 status consistency on every data row. The gate renders through
the same `render_curvature_process_plate` code path as the report.

Environment note: report plates were rendered with matplotlib **3.10.0**
(conda `base`). The `pinn` env carries 3.10.9, which produces ~0.36%
antialiasing pixel diffs — run the gate in `base`.

## Resolved run configs (verbatim; raw copies will be deleted)

None recoverable — the raw run dirs were already deleted before this provenance sidecar was (re)written, so no `config.resolved.json` is available to archive verbatim. Everything else in this MD/CSV is unaffected; only this one verbatim block is permanently empty.

---

# Stationary Ellipse (E1/E2) — source archive

**This CSV, this MD, and `stationary_ellipse_provenance.json` are the sole long-term
record for the ellipse report plates. Raw deletion is scoped and gated —
see "Deletion preconditions" below; never delete the raw root wholesale.**

- archive: `stationary_ellipse_source.csv.gz` — 206.2 MB gz, sha256 `780ac22544f774a806df2c0254f02840582c6b5215fa7f2451878e3d527c3bfa`
- built: 2026-07-07T16:41:14Z by `tem/report_source_archive/build_stationary_source.py --family ellipse`
- supersedes the retired `summary.csv` / `summary_index.csv` /
  `summary_manifest.json` / `summary_schema.md` (1.27 GB wide-format CSV).
- the csv.gz is gitignored (size); this MD records its sha256, and
  `verify_backup.py` checks external copies against it.

Load with: `pandas.read_csv("stationary_ellipse_source.csv.gz")` (pandas decompresses transparently).
Filter with `row_type`, then `case_id/grid_n/method/series`.

## CSV schema

One gzipped CSV, one header, four `row_type` values. Column groups (unused
groups are empty on a given row; gzip absorbs the cost):

| group | columns | used by |
|---|---|---|
| keys (all rows) | `row_type, case_id, grid_n, method, series, status, report_ready` | filtering |
| trace | `i, t, dt, tau, Umax, Ca, mass, kappa_linf, kappa_probe_linf, dc` | `row_type=trace` |
| curvature | `snapshot_index, snapshot_fraction, x, y, theta_deg, d, abs_d_over_delta, hk_native, hk_nn, delta_hk, native_kappa, kappa_nn, delta_kappa, kappa_nn_from_hk_over_delta, scale_identity_error, sign_product, hk_nn_raw, hk_nn_force, delta_hk_raw, delta_hk_force, relax_lambda, relax_neighbor_count` (+ shares `i, t, tau, Ca`) | `row_type=curvature` |
| diagnostic | curvature group's shared geometry columns + `theta, hk_analytic, kappa_analytic, delta_hk_native_analytic, delta_hk_nn_analytic, delta_kappa_native_analytic, delta_kappa_nn_analytic` | `row_type=diagnostic` (ellipse only) |
| index | `run_id, source_trace_csv, source_curvature_csv, benchmark, deployable, evidence_level, repeat_id, sigma, mu, expected_tmax, reached_final_time, final_t, final_tau, Ca_final, Ca_max, Ca_tail_mean, Ca_tail_max, mass_final, kappa_linf_tail_max, kappa_linf_tail_mean, kappa_probe_linf_tail_max, current_t, current_tau, current_Ca, trace_row_count, curvature_row_count, resolved_config_sha256, binary_sha256, official_source_manifest_sha256, geometry, report_png, note` | `row_type=index` |

### Row types

- `index` — one row per (case_id, grid_n, method, series) unit: provenance,
  completion status, evidence level, final metrics. All long strings live
  here only.
- `trace` — per-timestep solver trace, verbatim from the raw
  `surface_tension_trace.csv` / `stationary_trace.csv` (no float round-trip).
- `curvature` — per-interface-point curvature-process rows, verbatim from the
  raw `curvature_process.csv`. The FULL radial band is kept: rows with
  `abs_d_over_delta > 1` are NOT filtered and the 4x quadrant mirroring to
  0-360 deg is NOT applied — both are redraw-time derivations
  (`write_curvature_process_source_data` in
  `cfd_applications_cleanroom/cfd_apps/curvature_diagnostic_shared.py`).
- `diagnostic` — (ellipse) static curvature-diagnostic tier with analytic
  ellipse reference columns, verbatim from `curvature_field.csv`.

### Columns dropped from the raw traces, and their exact recovery

| dropped | recovery |
|---|---|
| `primary_metric_name` | constant `Ca` (asserted at build) |
| `primary_metric_value` | bitwise identical to `Ca` on every row (asserted at build) |
| `sigma, mu, benchmark, deployable, evidence_level, repeat_id` | constant per file (asserted at build); archived verbatim on the unit's index row |
| `run_id, case_id, method, level, grid_n` | constant per file; index-row keys. `level = log2(grid_n)` |

`Umax` and `tau` are kept verbatim (Umax = Ca*sigma/mu and tau = t/TMAX only
up to 1 ulp, so they are archived rather than derived).

### Semantics

- `status`: `complete` (reached final time, summary verified),
  `partial_trace_no_summary` (run did not finish; only current values),
  `complete_manifest` (diagnostic tier).
- `report_ready`: `false` rows must never be presented as final evidence.
- Tail metrics use the last 20% of trace rows (`tail = ca[int(0.8*len):]`),
  matching `stationary.py`. `Ca_final`/`Ca_max`/`Ca_tail_max` recompute
  bitwise from the trace rows; `Ca_tail_mean` matches to <1e-12 relative
  (summation-order noise: the run-time writer used a different sum order).

## Final metrics (from `row_type=index`; regenerate, don't hand-edit)

| case | N | method | series | status | ready | evidence | Ca_final | Ca_max | Ca_tail_mean | current_tau | rows |
|---|---:|---|---|---|---|---|---|---|---|---|---:|
| E1 | 64 | NN_DISABLE | main | complete | true | controlled_diagnostic | 2.503e-05 | 2.417e-03 | 2.503e-05 | 9.9999e-01 | 71133 |
| E1 | 64 | NN27_RAW | main | complete | true | controlled_diagnostic | 3.333e-05 | 2.415e-03 | 3.333e-05 | 9.9999e-01 | 71133 |
| E1 | 128 | NN_DISABLE | main | complete | true | controlled_diagnostic | 1.052e-05 | 2.401e-03 | 1.052e-05 | 1.0000e+00 | 201195 |
| E1 | 128 | NN27_RAW | main | complete | true | controlled_diagnostic | 1.388e-05 | 2.401e-03 | 1.388e-05 | 1.0000e+00 | 201195 |
| E1 | 256 | NN_DISABLE | main | complete | true | controlled_diagnostic | 6.308e-06 | 2.400e-03 | 6.360e-06 | 1.0000e+00 | 569064 |
| E1 | 256 | NN27_RAW | main | complete | true | controlled_diagnostic | 8.190e-06 | 2.400e-03 | 8.190e-06 | 1.0000e+00 | 569064 |
| E2 | 64 | NN_DISABLE | main | complete | true | controlled_diagnostic | 2.502e-05 | 4.656e-03 | 2.502e-05 | 9.9999e-01 | 71133 |
| E2 | 64 | NN27_RAW | main | complete | true | controlled_diagnostic | 3.332e-05 | 4.655e-03 | 3.332e-05 | 9.9999e-01 | 71133 |
| E2 | 128 | NN_DISABLE | main | complete | true | controlled_diagnostic | 1.052e-05 | 4.611e-03 | 1.052e-05 | 1.0000e+00 | 201195 |
| E2 | 128 | NN27_RAW | main | complete | true | controlled_diagnostic | 1.388e-05 | 4.610e-03 | 1.388e-05 | 1.0000e+00 | 201195 |
| E2 | 256 | NN_DISABLE | main | complete | true | controlled_diagnostic | 6.307e-06 | 4.610e-03 | 6.370e-06 | 1.0000e+00 | 569064 |
| E2 | 256 | NN27_RAW | main | complete | true | controlled_diagnostic | 8.191e-06 | 4.610e-03 | 8.192e-06 | 1.0000e+00 | 569064 |


## Provenance and interim state

- Complete units selected per `latest_complete_process_summary` (same rule as
  the retired merge script); partial units carry the newest local trace.
  Ellipse selection is intentionally dynamic — evidence accrues across run
  dirs as new ones are added (WSL import or local native rerun); determinism
  is provided by the per-unit `run_id` recorded on each index row (a rebuild
  that switches evidence shows up as a `run_id` diff in this MD's table
  source).
- **L6/L7/L8 core methods complete.** E1/E2 x {L6,L7,L8} x
  {NN_DISABLE, NN27_RAW} (12 units) are all `complete`/`report_ready=true`.
  L8 was imported from a WSL package (`run_and_pack_stationary_ellipse_l8_wsl.sh`
  + `import_stationary_ellipse_l8_package.sh`, 2026-07-07). L6/L7 were
  restored 2026-07-08 by re-running natively on this Mac (no WSL dependency;
  see `basilisk-qcc-compile-fragility` project memory) after the prior
  interim L6/L7 evidence was lost to an unpinned rebuild silently overwriting
  the archive in place — see the incident note in the
  `report-source-archive-protocol` project memory.
- Missing locally: E1/L6/NN_PROBE_ONLY, E1/L6/NN27_D4, E1/L7/NN_PROBE_ONLY, E1/L7/NN27_D4, E1/L8/NN_PROBE_ONLY, E1/L8/NN27_D4, E2/L6/NN_PROBE_ONLY, E2/L6/NN27_D4, E2/L7/NN_PROBE_ONLY, E2/L7/NN27_D4, E2/L8/NN_PROBE_ONLY, E2/L8/NN27_D4, diagnostic:E1/L6/NN27_RAW, diagnostic:E1/L6/NN27_D4, diagnostic:E1/L7/NN27_RAW, diagnostic:E1/L7/NN27_D4, diagnostic:E1/L8/NN27_RAW, diagnostic:E1/L8/NN27_D4, diagnostic:E2/L6/NN27_RAW, diagnostic:E2/L6/NN27_D4, diagnostic:E2/L7/NN27_RAW, diagnostic:E2/L7/NN27_D4, diagnostic:E2/L8/NN27_RAW, diagnostic:E2/L8/NN27_D4 — all
  `NN_PROBE_ONLY`/`NN27_D4` (never attempted for this restoration; out of
  scope) plus the ellipse `diagnostic` tier for RAW/D4 (a separate,
  unrelated tier, see the `diagnostic` row_type in the schema above).
- **Known loss (2026-07-06): `E1/L6/NN27_D4` process-tier complete evidence
  was pruned from local raw** (the 2026-07-06 status counted it in 16/24;
  the run dirs referenced by `stationary_ellipse_open_rows.csv`, e.g.
  `*_181444Z`, no longer exist). Only a 3,371-row abandoned trace remains
  and is archived as `partial_trace_no_summary`. The diagnostic tier for
  E1/L6/NN27_D4 IS complete and archived. Restore requires the WSL package
  or a re-run — do not mark this unit complete without new evidence. This is
  unrelated to the NN_DISABLE/NN27_RAW units, which are all complete.
- Report plates `stationary_ellipse_{E1,E2}_N{64,128,256}.png` all exist;
  the N64/N128 pair are the `partial_ready` plates (NN27_RAW curvature/trace
  + NN_DISABLE native trace; log Ca axis, snapshot-scoped 1-99% quantile
  ylims), N256 uses the same rendering path against the L8 units.

## Irrecoverable boundary after raw deletion

Archived verbatim: per-step traces for every locally-present unit,
curvature-process rows for complete units, diagnostic-tier
`curvature_field.csv` rows (12/12 with analytic reference), evidence
levels, config JSONs, sha256s.

NOT archived — gone forever once the archived ellipse run dirs are deleted:
- `build/` trees, logs, `status.json`, `source_hashes.json`
- diagnostic-tier `curvature_trace.csv` stubs (2 lines, no data)
- anything from the already-pruned run dirs (`*_181444Z` etc.) — those were
  deleted before this archive existed and are only recoverable from the WSL
  side, if at all.

## Deletion preconditions (ALL must hold; none are optional)

1. **External backup verified** — the csv.gz files are gitignored (the
   ellipse one exceeds GitHub's 100 MB limit), so an out-of-repo backup must
   exist and verify before any raw deletion:
   `python3 tem/report_source_archive/verify_backup.py <backup_dir>`
   (recomputes sha256 of the backup copies against the local archives and
   the hashes recorded in these MDs).
2. **Redraw gate PASS** (commands below).
3. **Scope audit** — deletion covers ONLY the raw run dirs this archive
   records in its index rows. Other dirs under the same raw root are NOT
   covered and need their own audit (see the irrecoverable-boundary section).
4. **`build_stationary_source.py --family ellipse` does not pin run ids**
   (unlike circle) — every rebuild rescans `results/raw/stationary_ellipse/`
   fresh and silently overwrites this archive in place with whatever's on
   disk. Before EVER deleting ellipse raw, re-verify precondition 1 (a
   fresh `verify_backup.py` PASS) — a stale backup from before the last
   rebuild is not sufficient. This is how the 2026-07-07 interim L6/L7
   evidence was lost the first time.

```bash
cd /Users/jcy/research/PINN
# rebuild archive from raw (only while raw still exists)
python3 tem/report_source_archive/build_stationary_source.py --family ellipse
# regenerate this MD (reads only csv.gz + tracked provenance json)
python3 tem/report_source_archive/gen_summary_md.py
# hard gate: redraw all report plates from the csv.gz ALONE and compare
PYTHONPATH=. /opt/anaconda3/bin/conda run -n base --no-capture-output \
  python tem/report_source_archive/gate_stationary_redraw.py --family ellipse
```

Gate checks: G1 byte-identical plate redraw from the archive only; G2 index
metrics recomputed from archived trace rows; G3 unit/row-count/snapshot
coverage; G4 status consistency on every data row. The gate renders through
the same `render_curvature_process_plate` code path as the report.

Environment note: report plates were rendered with matplotlib **3.10.0**
(conda `base`). The `pinn` env carries 3.10.9, which produces ~0.36%
antialiasing pixel diffs — run the gate in `base`.

## Resolved run configs (verbatim; raw copies will be deleted)

### `cfd_applications_cleanroom/results/raw/stationary_ellipse/stationary_ellipse_curvature_process_E1_20260707T154120Z/config.resolved.json`

```json
{
  "a": 0.4472136,
  "b": 0.3577709,
  "benchmark": "stationary_ellipse",
  "case": "E1",
  "case_source": "cfd_applications_cleanroom/cases/stationary_ellipse/stationary_ellipse_curvature_process.c",
  "diameter_equiv": 0.8000000304105994,
  "expected_tmax": 78.38367623848882,
  "levels": [
    6
  ],
  "methods": [
    "NN_DISABLE"
  ],
  "not_a_stationary_bubble_gate": true,
  "snapshot_fractions": [
    0.0,
    0.3333333333333333,
    0.6666666666666666,
    1.0
  ],
  "tier": "curvature-process"
}
```

### `cfd_applications_cleanroom/results/raw/stationary_ellipse/stationary_ellipse_curvature_process_E1_20260707T154357Z/config.resolved.json`

```json
{
  "a": 0.4472136,
  "b": 0.3577709,
  "benchmark": "stationary_ellipse",
  "case": "E1",
  "case_source": "cfd_applications_cleanroom/cases/stationary_ellipse/stationary_ellipse_curvature_process.c",
  "diameter_equiv": 0.8000000304105994,
  "expected_tmax": 78.38367623848882,
  "levels": [
    6,
    7
  ],
  "methods": [
    "NN_DISABLE",
    "NN27_RAW"
  ],
  "not_a_stationary_bubble_gate": true,
  "snapshot_fractions": [
    0.0,
    0.3333333333333333,
    0.6666666666666666,
    1.0
  ],
  "tier": "curvature-process"
}
```

### `cfd_applications_cleanroom/results/raw/stationary_ellipse/stationary_ellipse_curvature_process_E2_20260707T154358Z/config.resolved.json`

```json
{
  "a": 0.4898979,
  "b": 0.3265986,
  "benchmark": "stationary_ellipse",
  "case": "E2",
  "case_source": "cfd_applications_cleanroom/cases/stationary_ellipse/stationary_ellipse_curvature_process.c",
  "diameter_equiv": 0.799999920707346,
  "expected_tmax": 78.38366011546945,
  "levels": [
    6,
    7
  ],
  "methods": [
    "NN_DISABLE",
    "NN27_RAW"
  ],
  "not_a_stationary_bubble_gate": true,
  "snapshot_fractions": [
    0.0,
    0.3333333333333333,
    0.6666666666666666,
    1.0
  ],
  "tier": "curvature-process"
}
```
