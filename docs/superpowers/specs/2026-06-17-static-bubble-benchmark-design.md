# Static Bubble Benchmark Design

Date: 2026-06-17
Topic: `cfd_static_bubble` stationary-bubble spurious-current benchmark

## 1. Objective

Implement the plan in [cfd_static_bubble/static_bubble_benchmark.md](/Users/jcy/research/PINN/cfd_static_bubble/static_bubble_benchmark.md) as an isolated benchmark package under `cfd_static_bubble/`, then run and verify the resulting artifacts without perturbing the existing training, evaluation, or output layout of the PINN repo.

The benchmark compares five curvature modes in one Basilisk solver:

- `EXACT`
- `HF33`
- `HF35`
- `NN27_RAW`
- `NN27_D4`

The benchmark must preserve the plan's scientific boundary:

- compare modes against each other in the same Basilisk setup
- treat `EXACT` as the Basilisk floor diagnostic, not a paper-number reproduction target
- use the current NN27 checkpoints and feature contract exactly as trained
- keep all benchmark outputs inside `cfd_static_bubble/`

## 2. Scope

### In Scope

- A self-contained `cfd_static_bubble` benchmark implementation
- C/Basilisk solver sources for the static-bubble cases
- Python export and verification scripts for the NN27 model
- Per-run raw CSV artifacts and a benchmark summary table
- Verification scripts and reports proving which planned gates passed

### Out of Scope

- Changing existing `evaluate/`, `train_generate/`, `testdata_generate/`, or `model/` interfaces
- Writing benchmark artifacts into `out/`, `dataset/`, or `test_data/`
- Vendoring Basilisk or other third-party solver code into the repo
- Reproducing OpenFOAM-identical paper magnitudes
- Implementing deferred v2 items from the benchmark plan

## 3. Assumptions

- Basilisk is installed outside this repo and will be provided via an environment variable or script argument such as `BASILISK_ROOT`.
- Existing repo checkpoints under `out/<N>/baseline_<N>_hgradient.pt` and normalization sidecars remain the source of truth for NN27.
- The current feature contract remains `input_dim=27`, `feature_order="phi9+nx9+ny9"`, `transform_kind="standardize"`.
- Existing repo code is used read-only as the truth source for feature extraction, standardization, and checkpoint loading.

If any assumption fails at runtime, the benchmark must fail loudly or skip only the affected NN rows with explicit status, never silently fall back to another contract.

## 4. Isolation Boundary

All implementation, generated headers, binaries, runtime results, and verification reports live under:

- [cfd_static_bubble/basilisk](/Users/jcy/research/PINN/cfd_static_bubble/basilisk)
- [cfd_static_bubble/scripts](/Users/jcy/research/PINN/cfd_static_bubble/scripts)
- [cfd_static_bubble/results](/Users/jcy/research/PINN/cfd_static_bubble/results)

No benchmark code is added to `evaluate/` unless there is no isolated alternative. Default behavior is:

- import existing Python helpers from current modules when useful
- do not move benchmark logic into existing repo packages
- do not reuse existing output directories for benchmark artifacts

This keeps the benchmark as a subproject instead of a cross-cutting repo refactor.

## 5. Target File Layout

### Solver sources

- `cfd_static_bubble/basilisk/static_bubble.c`
- `cfd_static_bubble/basilisk/ml_curvature.h`
- `cfd_static_bubble/basilisk/hf_curvature.h`
- `cfd_static_bubble/basilisk/mlp_forward.h`

### Generated solver inputs

- `cfd_static_bubble/basilisk/mlp_weights.h`
- `cfd_static_bubble/basilisk/parity_fixtures.h`

These are generated files, but they remain inside the benchmark subtree so the rest of the repo stays untouched.

### Scripts

- `cfd_static_bubble/scripts/export_mlp_to_c.py`
- `cfd_static_bubble/scripts/run_matrix.sh`
- `cfd_static_bubble/scripts/summarize_results.py`
- `cfd_static_bubble/scripts/verify_results.py`

### Runtime artifacts

- `cfd_static_bubble/results/runs/<case>/N<rho>/<mode>/...`
- `cfd_static_bubble/results/summary.csv`
- `cfd_static_bubble/results/verification.json`
- `cfd_static_bubble/results/verification.md`

## 6. Architecture

The benchmark has four layers.

### 6.1 Model export layer

`export_mlp_to_c.py` reads the existing PyTorch checkpoint and its normalization data, asserts the expected NN27 contract, then emits:

- `mlp_weights.h` for C inference
- `parity_fixtures.h` for exact PyTorch-vs-C checks

The export layer is the only writer of generated NN headers. The solver never reads the PyTorch checkpoint directly.

### 6.2 Solver layer

`static_bubble.c` owns:

- domain geometry
- case selection (`QUAD`, `FULL`)
- grid selection (`N`)
- fluid parameters
- time stepping
- runtime diagnostics
- CSV emission

`ml_curvature.h` owns curvature assignment on the shared interface mask `M`.

`hf_curvature.h` owns HF33 and HF35, with an explicit VOF-only contract and explicit stencil support.

`mlp_forward.h` owns pure-C standardized forward inference for the exported NN27 model.

### 6.3 Sweep layer

`run_matrix.sh` orchestrates build-and-run sequencing:

1. `EXACT`
2. `HF33` and `HF35`
3. `NN27_RAW` and `NN27_D4`

It starts with `N=64` and `N=128`, then advances to `256` and `512` only after the smaller grids produce valid results. This preserves the full target matrix while reducing wasted runtime on broken intermediate states.

### 6.4 Verification layer

`verify_results.py` is the authoritative completion checker for the benchmark artifact set. It verifies:

- expected run coverage
- metric-column completeness
- metadata completeness
- gate pass/fail state
- explicit skipped or failed rows

It produces machine-readable and human-readable reports so completion claims are backed by files, not by terminal narration.

## 7. Data and Control Flow

The end-to-end flow is:

1. Read `baseline_<N>_hgradient.pt/.csv`
2. Assert the NN27 feature contract
3. Export C weights and parity fixtures
4. Compile Basilisk solver for one `(CASE, N, MODE)` combination
5. Run the case and emit a per-run CSV plus metadata
6. Reduce all valid runs into `summary.csv`
7. Verify summary and run inventory against the benchmark requirements

Mode-specific input paths are fixed:

- `EXACT`: `kappa = 1/R` on the shared mask `M`
- `HF33`, `HF35`: curvature from VOF `f` only
- `NN27_RAW`, `NN27_D4`: curvature from analytic SDF-based 27-D input only

Comparable modes must all share the same force path:

- cell-centered `kappa`
- `phi_pot = sigma * kappa`
- `iforce.h` application

`tension.h` may only appear in clearly labeled diagnostic-only reference runs, never in core rows.

## 8. Runtime Contracts

These are hard rules for the implementation.

### 8.1 External dependency contract

- The benchmark does not assume `qcc` is globally on `PATH`.
- Scripts accept `BASILISK_ROOT` or fail with an actionable error.
- Missing Basilisk is a benchmark-environment failure, not a signal to rewrite the benchmark around another solver.

### 8.2 NN contract

- The exported checkpoint must be a 27-D V2 checkpoint.
- `pca18` and other incompatible checkpoints are rejected.
- Standardization is applied exactly once, in C, matching PyTorch.
- C parity must pass before any NN solver run is accepted.

### 8.3 HF contract

- HF33 reads no more than a 3x3 support region.
- HF35 reads no more than a 3x5 support region.
- Both consume VOF `f`, not analytic `phi`.
- Both use the shared force path for core comparisons.

### 8.4 Result contract

Every accepted run emits:

- a per-sample CSV with the planned runtime columns
- run metadata containing case, `N`, mode, checkpoint or `EXACT`, force path, curvature path, `epsilon`, `R`, `La`, and status

No row enters `summary.csv` unless the run passes the benchmark's validity checks.

## 9. Error Handling

### Fail-fast errors

These stop the relevant step immediately:

- missing `BASILISK_ROOT` or unusable solver headers
- checkpoint contract mismatch
- parity failure
- malformed runtime CSV
- missing required metadata

### Explicit skips

These are allowed only when they are isolated and reported:

- missing NN checkpoint for one `N`
- large-grid runs that are still pending during phased execution

Skipped or pending rows must be recorded explicitly, not silently absent. Pending large-grid rows are acceptable only as an intermediate execution state; they do not count as satisfied deliverables in the final completion audit.

### Invalid-but-recorded runs

If a run completes but fails a benchmark gate, it remains on disk but is marked invalid for summary inclusion. Examples:

- plateau window not flat enough
- `Ca_eq` not finite
- `f_drift` too large for interpretation
- repeatability failure for `EXACT`

## 10. Verification Strategy

Verification is divided into three stages.

### 10.1 Static verification

- Exporter asserts checkpoint structure and transform contract
- C parity check requires max relative error `< 1e-6`
- D4 symmetric-input identity test
- HF stencil-bound and basic-sign tests

If static verification fails, later stages must not claim benchmark success.

### 10.2 Runtime verification

- `EXACT` repeatability check at `N=64`
- plateau-window validity for every accepted run
- finite `Ca_eq`
- required runtime columns present
- consistent metadata present

### 10.3 Artifact verification

`verify_results.py` checks the benchmark deliverables from the plan:

- two cases: `QUAD`, `FULL`
- target resolutions: `64`, `128`, `256`, `512`
- five core modes
- summary metrics: `Ca_eq`, `E2`, `Einf`, `kappa_mean`, `kappa_std`, `f_drift`
- explicit accounting of completed, skipped, and invalid rows

The report distinguishes:

- requirement satisfied
- requirement partially satisfied
- requirement missing
- requirement blocked by external dependency or runtime cost

## 11. Implementation Order

The implementation order is fixed so later work cannot bypass earlier gates.

1. Create isolated benchmark scaffolding under `cfd_static_bubble/`
2. Implement `export_mlp_to_c.py`
3. Implement and verify C forward parity
4. Implement `EXACT`
5. Implement and verify `HF33` and `HF35`
6. Implement and verify `NN27_RAW`
7. Implement and verify `NN27_D4`
8. Implement run orchestration and summarization
9. Run the matrix in phases `64 -> 128 -> 256 -> 512`
10. Generate and review final verification artifacts

This order matches the benchmark document's gate sequence and prevents premature NN claims.

## 12. Risks and Mitigations

### External Basilisk path drift

Mitigation:

- require explicit `BASILISK_ROOT`
- keep build logic in one script
- fail with a direct message naming the missing path or header

### Solver-force inconsistency across modes

Mitigation:

- keep a single core `iforce.h` path for all comparable rows
- label any native solver reference runs separately

### Feature-contract mismatch

Mitigation:

- reuse repo truth sources for feature extraction and transforms
- enforce parity before NN runtime use

### Large-grid runtime cost

Mitigation:

- phase runs from 64 to 512
- keep full matrix as the target
- record partial completion honestly if 256 or 512 remain unfinished at a given checkpoint

### Scope creep into the rest of the repo

Mitigation:

- keep every new file and every generated artifact under `cfd_static_bubble/` or the design-spec path
- default to importing shared helpers rather than refactoring existing modules

## 13. Acceptance Criteria

The benchmark is considered implemented only when the following are all true:

- the benchmark code exists under the isolated `cfd_static_bubble/` structure
- the exporter can produce C weights from repo checkpoints with contract checks
- parity verification proves the C NN path matches PyTorch within the specified tolerance
- the Basilisk solver can run the planned modes under the shared force path
- raw result CSVs and metadata are written under `cfd_static_bubble/results/`
- `summary.csv` exists and contains valid accepted rows
- `verification.json` and `verification.md` explicitly assess the benchmark deliverables
- the verification artifacts account for every planned `case x resolution x mode` row as valid, invalid, skipped, or pending
- no planned row remains implicitly missing
- the final completion claim is backed by those verification artifacts

## 14. Review Notes

This design intentionally optimizes for isolation and auditability over clever reuse. The benchmark is a new solver-facing experiment, so the stable choice is to keep it as a bounded subproject that reads from the existing PINN assets but does not mutate their workflows or output surfaces.
