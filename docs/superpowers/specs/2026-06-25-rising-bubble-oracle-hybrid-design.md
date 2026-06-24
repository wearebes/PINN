# Rising Bubble NN Curvature Evidence Package

Date: 2026-06-25
Scope: Basilisk rising-bubble benchmark under `basilisk_reference_gate/rising/`.

## 1. Scientific Goal

Evaluate whether the current NN curvature family provides measurable improvement over the native Basilisk CLSVOF-LS curvature path in the Hysing Case 1 rising-bubble dynamic benchmark.

The goal is not to prove that an oracle selector is deployable. The oracle is only an upper-bound diagnostic for the NN candidate family. If the upper bound does not improve over the native reference, this case should stop rather than accumulating more figures and model variants.

## 2. Current Repo Facts

The current accepted native rising-bubble reference is `REF_BG2`, implemented through the Phase 2 rising host in:

- `basilisk_reference_gate/rising/cases/rising_bubble_nn.c`
- `basilisk_reference_gate/rising/scripts/run_phase2_nn.sh`
- `basilisk_reference_gate/rising/results/reports/phase2_nn_summary.md`

Existing accepted Phase 2 evidence says:

- `REF_BG2`, `NN_RAW`, and `NN_D4` all reach `t=3`.
- `ref_bg2 vs nn_disable` parity passes.
- `ref_bg2 vs nn_probe` parity passes.
- `NN_RAW/NN_D4` are interpretable, but current dynamic metrics are very close to `REF_BG2` and mostly slightly worse except small mass-drift differences.

Important implementation fact:

`distance_curvature()` in `basilisk_reference_gate/nn/integral_nn.h` computes the level-set curvature by 3x3 central differences:

```text
dx  = (d[1] - d[-1]) / 2
dy  = (d[0,1] - d[0,-1]) / 2
dxx = d[1] - 2*d[] + d[-1]
dyy = d[0,1] - 2*d[] + d[0,-1]
dxy = (d[1,1] - d[-1,1] - d[1,-1] + d[-1,-1]) / 4
kappa = (dx^2*dyy - 2*dx*dy*dxy + dy^2*dxx) / |grad d|^3 / Delta
```

Therefore, if a proposed `hk_cd` uses this same stencil and the same `d` field, then:

```text
hk_cd == hk_ref
```

up to floating-point roundoff. In that case, central-difference curvature is not an independent candidate and must not enter an oracle hybrid selector.

## 3. Method Matrix

| ID | Curvature source | Deployable | Purpose |
|---|---|---:|---|
| `REF_BG2` | Native Basilisk CLSVOF-LS `distance_curvature()` | yes | Accepted reference baseline |
| `NN_RAW` | Direct NN curvature prediction | yes | Deployable NN replacement |
| `NN_D4` | D4-consensus NN curvature prediction | yes | Deployable symmetry-preserving NN replacement |
| `ORACLE_NN_BEST` | Per-point oracle choice between `NN_RAW` and `NN_D4` using `hk_ref` | no | Upper bound of the NN family |

Central-difference curvature is logged only as a diagnostic probe. It is excluded from the oracle selector if it is numerically reference-equivalent.

## 4. Hard Gates

### Gate 0: Compile, Status, And Existing Parity

Before interpreting any new output:

- Build from the project-local Basilisk rising host, not by editing Basilisk upstream.
- Preserve compile logs, status files, run command metadata, CSVs, and stderr probe logs.
- Reconfirm `REF_BG2` builds and reaches `t=3`.
- Reconfirm no-op parity: `REF_BG2` vs `NN_DISABLE`.
- Reconfirm probe-only parity: `REF_BG2` vs `NN_PROBE`.

If no-op parity fails, NN trajectory deltas are diagnostic only.

### Gate 1: CD-vs-REF Non-Degeneracy

At runtime, record:

- `hk_ref = Delta * distance_curvature(point, d)`
- `hk_cd`
- `hk_nn_raw`
- `hk_nn_d4`

For the central-difference probe:

```text
e_cd = hk_cd - hk_ref
```

Report:

- `max_abs(e_cd)`
- `mean_abs(e_cd)`
- `rms(e_cd)`
- `p99_abs(e_cd)`
- `nonzero_count`
- threshold decisions at `1e-12`, `1e-10`, and `1e-8`

Decision rule:

```text
max_abs(e_cd) <= 1e-12
=> CD is reference-equivalent and excluded from oracle candidates.
```

If `hk_cd` unexpectedly differs from `hk_ref`, stop and inspect the formula, field source, scaling, and sample timing before claiming any hybrid result.

### Gate 2: Runtime Validity

For every accepted trajectory:

- no NaN/Inf before `t=3`
- final time reaches `t=3`
- mass drift is reported
- `yc(t)`, `vc(t)`, and `circ(t)` are available
- compare `yc`, `vc`, and `circ` against MooNMD where applicable
- compare all non-reference methods against `REF_BG2`

### Gate 3: Oracle Interpretation

`ORACLE_NN_BEST` must be labeled as:

```text
non-deployable upper bound
```

It cannot be described as a solver method or as a deployable hybrid.

## 5. Oracle Definition

For every probed interface point:

```text
hk_oracle =
  hk_raw, if |hk_raw - hk_ref| <= |hk_d4 - hk_ref|
  hk_d4,  otherwise
```

Tie-breaking chooses `NN_RAW`.

Central/native is not a selectable candidate. The oracle means:

```text
In the ideal case where true native curvature is known, how much useful local
curvature information exists inside the NN candidate family?
```

It does not mean:

```text
Native + NN deployable hybrid
```

## 6. Evidence Outputs

### Raw Outputs To Preserve

- trajectory CSV files
- curvature-probe CSV files
- status files
- compile logs
- stderr probe logs
- run command logs, if available
- git hash or local metadata, if available

### Final Tables

One trajectory comparison table:

| Method | mass drift | vc_max | yc(t=3) | circ_min | delta vc vs REF | delta yc vs REF | delta circ vs REF |
|---|---:|---:|---:|---:|---:|---:|---:|
| `REF_BG2` | | | | | | | |
| `NN_RAW` | | | | | | | |
| `NN_D4` | | | | | | | |
| `ORACLE_NN_BEST` | | | | | | | |

One curvature-probe table:

| Probe | max_abs | mean_abs | rms | p99_abs | nonzero_count |
|---|---:|---:|---:|---:|---:|
| `hk_cd - hk_ref` | | | | | |
| `hk_nn_raw - hk_ref` | | | | | |
| `hk_nn_d4 - hk_ref` | | | | | |
| `hk_oracle - hk_ref` | | | | | |

### Final Figures

Use a clean final figure set only:

- `yc(t)` comparison
- `vc(t)` comparison
- `circ(t)` comparison
- curvature error CDF or histogram for `|hk_m - hk_ref|`
- optional scatter: `hk_candidate` vs `hk_ref`

## 7. Cleanup And Archival Rules

Accepted evidence only includes:

- `REF_BG2`
- `NN_RAW`
- `NN_D4`
- `ORACLE_NN_BEST`

Archive or isolate from the final report:

- old imax sweeps
- PCA variants
- redistance sensitivity plots
- obsolete central-diff oracle attempts
- failed runs
- non-parity runs
- generated `.qcc*` expansion directories
- scratch directories, after accepted raw outputs are preserved

Archived material may remain reproducible, but it must not be mixed into the main conclusion.

## 8. Decision Logic

### Case A: Deployable NN Improves REF

If `NN_RAW` or `NN_D4` improves dynamic metrics over `REF_BG2` without violating gates:

```text
Deployable NN curvature replacement provides measurable improvement in this host test.
```

### Case B: Only Oracle Improves REF

If `ORACLE_NN_BEST` improves `REF_BG2`, but `NN_RAW` and `NN_D4` do not:

```text
The NN family contains useful curvature information, but a deployable
selection/gating rule is still missing.
```

### Case C: Oracle Does Not Improve REF

If `ORACLE_NN_BEST` does not improve `REF_BG2`:

```text
The current NN curvature family has no exploitable advantage in this Basilisk
rising-bubble host test.
```

This should stop further model/sweep accumulation for this case unless the scientific question changes.

## 9. Implementation Prompt For A New Window

Copy the prompt below into a new Codex window.

```markdown
You are working on the Basilisk rising-bubble NN curvature evidence package in `/Users/jcy/research/PINN`.

Read these files first:

- `docs/superpowers/specs/2026-06-25-rising-bubble-oracle-hybrid-design.md`
- `basilisk_reference_gate/docs/rising_bubble_plan.md`
- `basilisk_reference_gate/rising/PHASE2_NN_BASELINE_IMPLEMENTATION_PLAN.md`
- `basilisk_reference_gate/rising/cases/rising_bubble_nn.c`
- `basilisk_reference_gate/nn/integral_nn.h`
- `basilisk_reference_gate/nn/nn_curvature.h`
- `basilisk_reference_gate/nn/nn_features.h`
- `basilisk_reference_gate/rising/scripts/run_phase2_nn.sh`
- `basilisk_reference_gate/rising/scripts/summarize_phase2_nn.py`
- `basilisk_reference_gate/rising/results/reports/phase2_nn_summary.md`

Important context:

- Current native reference is `REF_BG2`.
- `distance_curvature()` in `integral_nn.h` is itself a 3x3 central-difference formula on the level-set field `d`.
- Therefore, central-difference curvature must not be treated as an independent hybrid candidate unless a runtime non-degeneracy probe proves it differs from `hk_ref`.
- This task uses oracle hybrid type A, but only as a non-deployable diagnostic upper bound.

Hard requirements:

1. Do not modify Basilisk upstream source.
2. Keep edits inside the project-local `basilisk_reference_gate/rising/` and `basilisk_reference_gate/nn/` layer unless a better local boundary already exists.
3. Preserve accepted reference evidence. Do not delete old outputs blindly; archive or isolate them only after preserving accepted raw files.
4. Do not claim `ORACLE_NN_BEST` is deployable.
5. Do not include central/native in the oracle selector.

Implementation tasks:

1. Reconfirm current Phase 2 baseline:
   - `REF_BG2` reaches `t=3`
   - `NN_RAW` reaches `t=3`
   - `NN_D4` reaches `t=3`
   - `REF_BG2` vs `NN_DISABLE` parity passes
   - `REF_BG2` vs `NN_PROBE` parity passes

2. Implement a curvature probe that records interface-band samples at controlled times:
   - `t in {0, 0.5, 1, 1.5, 2, 2.5, 3}` if practical
   - otherwise use a documented representative subset
   - record `t`, `i`, cell/sample coordinates or sample index, `Delta`, `d`, `f`, `hk_ref`, `hk_cd`, `hk_nn_raw`, `hk_nn_d4`

3. Implement `CD-vs-REF non-degeneracy gate`:
   - compute `e_cd = hk_cd - hk_ref`
   - report `max_abs`, `mean_abs`, `rms`, `p99_abs`, `nonzero_count`
   - report conclusions under thresholds `1e-12`, `1e-10`, `1e-8`
   - if `max_abs <= 1e-12`, declare CD reference-equivalent and exclude it from oracle candidates
   - if CD unexpectedly differs, stop and explain why before running oracle conclusions

4. Implement oracle only over NN candidates:
   - `ORACLE_NN_BEST = NN_RAW` if `|hk_raw - hk_ref| <= |hk_d4 - hk_ref|`
   - otherwise `ORACLE_NN_BEST = NN_D4`
   - central/native must remain baseline only

5. Produce clean evidence package:
   - trajectory comparison table for `REF_BG2`, `NN_RAW`, `NN_D4`, `ORACLE_NN_BEST`
   - curvature-probe table for `hk_cd`, `hk_nn_raw`, `hk_nn_d4`, `hk_oracle` errors against `hk_ref`
   - figures for `yc(t)`, `vc(t)`, `circ(t)`, and curvature-error distributions
   - final Markdown report that states exactly which gates passed or failed

6. Cleanup/archival:
   - keep accepted raw CSVs, status files, compile logs, and probe CSVs
   - archive or isolate old imax/PCA/redistance/sweep figures from the main final report
   - remove generated scratch or `.qcc*` directories only if doing so does not destroy accepted evidence

Stop and report if:

- parity with `REF_BG2` fails
- any method has NaN/Inf before `t=3`
- `CD-vs-REF` shows unexpected nonzero mismatch
- oracle implementation accidentally includes central/native as a selectable candidate

Final response must include:

- exact files changed or created
- exact commands run
- gate results
- whether the outcome is Case A, Case B, or Case C from the design doc
- a short statement that `ORACLE_NN_BEST` is non-deployable upper-bound evidence only
```

## 10. Self-Review

- No placeholder requirements remain.
- `central/native` is excluded from the oracle selector when reference-equivalent.
- The oracle is labeled non-deployable in every relevant section.
- Cleanup preserves provenance rather than deleting evidence blindly.
- Scope is one implementation plan: probe, oracle, clean report, and archive/isolate old outputs.
