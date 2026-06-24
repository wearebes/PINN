# Rising Bubble NN Curvature Diagnostic and Evidence Package

Date: 2026-06-25
Scope: Basilisk rising-bubble benchmark under `basilisk_reference_gate/rising/`.

## 1. Scientific Goal

Evaluate two separate questions in the Hysing Case 1 rising-bubble dynamic benchmark:

1. Whether deployable NN curvature replacements improve the dynamic benchmark metrics over the native Basilisk CLSVOF-LS curvature path.
2. Whether the current NN family contains useful curvature information, diagnosed separately by native-mimic local probes and non-deployable trajectory-level oracle comparisons.

The local oracle based on native curvature is not a physical-truth oracle. It only tests whether the NN family can mimic the native Basilisk curvature on the same state. Improvement over native must be judged against external benchmark quantities, primarily MooNMD/Hysing metrics, with an effect-size gate above parity noise.

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

Here:

```text
hk_ref = Delta * distance_curvature(point, d)
```

is native Basilisk curvature, not exact physical curvature. It is valid as a reference label for native-mimic diagnostics, but it cannot define whether NN curvature is better than native in the physical benchmark.

If a proposed `hk_cd` uses this same stencil and the same `d` field, then:

```text
hk_cd == hk_ref
```

up to floating-point roundoff. In that case, central-difference curvature is not an independent candidate and must not enter an oracle hybrid selector.

## 3. Method Matrix

| ID | Definition | Deployable | What It Can Support |
|---|---|---:|---|
| `REF_BG2` | Native Basilisk CLSVOF-LS `distance_curvature()` | yes | Accepted native baseline |
| `NN_RAW` | Direct NN curvature prediction | yes | Deployable NN replacement |
| `NN_D4` | D4-consensus NN curvature prediction | yes | Deployable symmetry-preserving NN replacement |
| `ORACLE_NATIVE_MIMIC` | Pointwise choose the NN candidate closer to native `hk_ref` | no | Whether the NN family contains native-like local curvature information |
| `ORACLE_DYNAMIC_BEST` | Hindsight choose the better complete NN trajectory by external MooNMD/Hysing score | no | Whether the NN family has trajectory-level potential against the external benchmark |

Central-difference curvature is logged only as a diagnostic probe. It is excluded from any oracle selector if it is numerically reference-equivalent.

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

`hk_cd` must be computed by an independent manual central-difference helper. It must not call `distance_curvature()`, otherwise the gate only proves code reuse tautology.

The helper should implement the same formula explicitly:

```c
static inline double manual_cd_hk (Point point, scalar d) {
  double dx  = (d[1] - d[-1])/2.;
  double dy  = (d[0,1] - d[0,-1])/2.;
  double dxx = d[1] - 2.*d[] + d[-1];
  double dyy = d[0,1] - 2.*d[] + d[0,-1];
  double dxy = (d[1,1] - d[-1,1] - d[1,-1] + d[-1,-1])/4.;
  double dn = sqrt(dx*dx + dy*dy) + 1e-30;
  double kappa = (dx*dx*dyy - 2.*dx*dy*dxy + dy*dy*dxx)
               / (dn*dn*dn) / Delta;
  return Delta * kappa;
}
```

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

### Gate 2: Deterministic Probe Sampling

Probe samples must be comparable within a trajectory:

- `hk_ref`, `hk_cd`, `hk_nn_raw`, and `hk_nn_d4` are written for the same cells at the same time.
- Smoke runs record all interface-band samples.
- Full runs record all samples if the eligible count is `<= 50000` at an output time.
- If the eligible count is larger, use deterministic stride or deterministic reservoir sampling capped at `50000` samples per output time.
- Record both `eligible_count` and `written_count`.

Recommended probe times:

```text
t in {0, 0.5, 1, 1.5, 2, 2.5, 3}
```

If exact event timing is impractical, use a documented deterministic nearest-step rule.

### Gate 3: Runtime Validity

For every accepted trajectory:

- no NaN/Inf before `t=3`
- final time reaches `t=3` within output tolerance
- circularity remains finite and positive
- mass drift is reported and satisfies:

```text
|mass_drift| <= max(2 * |mass_drift_REF|, absolute_mass_tol)
```

`absolute_mass_tol` must be derived from the already accepted Phase 2 evidence scale, not guessed.

- `yc(t)`, `vc(t)`, and `circ(t)` are available
- compare `yc`, `vc`, and `circ` against MooNMD where applicable
- compare all non-reference methods against `REF_BG2`

### Gate 4: External-Metric Effect Size

Define a trajectory-level external score `J_m` using MooNMD/Hysing quantities, not `REF_BG2` as truth:

```text
J_m =
  w_v * |vc_max_m - vc_max_MooNMD| / s_v
  + w_y * |yc_t3_m - yc_t3_MooNMD| / s_y
  + w_c * |circ_min_m - circ_min_MooNMD| / s_c
  + w_M * |mass_drift_m| / s_M
```

The implementation must define and report the weights and scales. A minimal first pass can use equal weights and scales from the Hysing acceptance bands, but the report must state that choice.

Improvement requires:

```text
J_candidate < J_REF - epsilon_effect
```

where:

```text
epsilon_effect = max(10 * parity_noise_floor, 0.05 * J_REF)
```

If the observed gain is below this threshold, report numerical equivalence rather than improvement.

### Gate 5: Oracle Interpretation

`ORACLE_NATIVE_MIMIC` and `ORACLE_DYNAMIC_BEST` must both be labeled as non-deployable diagnostics.

`ORACLE_NATIVE_MIMIC` cannot be used as evidence of improvement over native because its local label is native curvature itself.

## 5. Oracle Definitions

### 5.1 `ORACLE_NATIVE_MIMIC`

For every probed interface point on a given trajectory state:

```text
hk_mimic =
  hk_raw, if |hk_raw - hk_ref| <= |hk_d4 - hk_ref|
  hk_d4,  otherwise
```

Tie-breaking chooses `NN_RAW`.

Central/native is not a selectable candidate. This oracle means:

```text
Which NN candidate is most native-like at this local stencil?
```

It does not mean:

```text
Which NN candidate is closer to exact physical curvature?
```

If an online native-mimic oracle is ever implemented as a solver trajectory, the report must state:

```text
The online oracle uses native curvature on its own evolving state. Therefore,
hk_ref is the native curvature of the oracle state, not a frozen REF_BG2
trajectory label.
```

### 5.2 `ORACLE_DYNAMIC_BEST`

This is a run-level hindsight oracle, not a pointwise curvature selector:

```text
ORACLE_DYNAMIC_BEST = argmin over {NN_RAW, NN_D4} of J_m
```

where `J_m` is the external MooNMD/Hysing trajectory score from Gate 4.

This oracle can answer:

```text
Does the current NN candidate family contain a complete trajectory that is
closer to the external benchmark than REF_BG2 by a meaningful margin?
```

It cannot answer:

```text
Do we already have a deployable hybrid selection rule?
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

| Method | mass drift | vc_max | yc(t=3) | circ_min | J vs MooNMD | improvement vs REF | effect-size status |
|---|---:|---:|---:|---:|---:|---:|---|
| `REF_BG2` | | | | | | | |
| `NN_RAW` | | | | | | | |
| `NN_D4` | | | | | | | |
| `ORACLE_DYNAMIC_BEST` | | | | | | | |

One external-metric error table:

| Method | E_vc vs MooNMD | E_yc vs MooNMD | E_circ vs MooNMD | E_mass | score J |
|---|---:|---:|---:|---:|---:|
| `REF_BG2` | | | | | |
| `NN_RAW` | | | | | |
| `NN_D4` | | | | | |
| `ORACLE_DYNAMIC_BEST` | | | | | |

One curvature-probe table:

| Probe | max_abs | mean_abs | rms | p99_abs | nonzero_count |
|---|---:|---:|---:|---:|---:|
| `hk_cd - hk_ref` | | | | | |
| `hk_nn_raw - hk_ref` | | | | | |
| `hk_nn_d4 - hk_ref` | | | | | |
| `hk_native_mimic - hk_ref` | | | | | |

### Final Figures

Use a clean final figure set only:

- `yc(t)` comparison
- `vc(t)` comparison
- `circ(t)` comparison
- curvature error CDF or histogram for `|hk_m - hk_ref|`
- optional scatter: `hk_candidate` vs `hk_ref`
- external score bar chart for `J`

## 7. Cleanup And Archival Rules

Accepted evidence only includes:

- `REF_BG2`
- `NN_RAW`
- `NN_D4`
- `ORACLE_NATIVE_MIMIC`
- `ORACLE_DYNAMIC_BEST`

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

### Case A: Deployable NN Improves The External Benchmark

If `NN_RAW` or `NN_D4` satisfies:

```text
J_NN < J_REF - epsilon_effect
```

without violating gates:

```text
Deployable NN curvature replacement provides measurable improvement in this host test.
```

### Case B: Only Dynamic Oracle Improves

If `ORACLE_DYNAMIC_BEST` satisfies the effect-size gate but neither deployable NN method does:

```text
The NN candidate family contains trajectory-level useful signal, but a
deployable selector is missing.
```

### Case C: Only Native-Mimic Looks Good

If `ORACLE_NATIVE_MIMIC` is close to native but no deployable or dynamic oracle trajectory improves the external score:

```text
The NN family can reproduce native-like curvature locally, but this does not
constitute improvement over native.
```

### Case D: No Useful Signal

If neither deployable NN nor `ORACLE_DYNAMIC_BEST` improves over `REF_BG2` by the effect-size gate:

```text
Stop this rising-bubble route for the current NN family.
```

This should stop further model/sweep accumulation for this case unless the scientific question changes.

## 9. Implementation Prompt For A New Window

Copy the prompt below into a new Codex window.

```markdown
You are working on the Basilisk rising-bubble NN curvature diagnostic and evidence package in `/Users/jcy/research/PINN`.

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
- `hk_ref = Delta * distance_curvature(point, d)` is native Basilisk curvature, not exact physical curvature.
- Any oracle that chooses `NN_RAW/NN_D4` by closeness to `hk_ref` is a native-mimic diagnostic only. It must not be interpreted as evidence that NN improves over native.
- To claim improvement over native, compare complete trajectories against external benchmark quantities such as MooNMD/Hysing metrics, and require an effect-size gate above parity noise.
- This task separates `ORACLE_NATIVE_MIMIC` from `ORACLE_DYNAMIC_BEST`.

Hard requirements:

1. Do not modify Basilisk upstream source.
2. Keep edits inside the project-local `basilisk_reference_gate/rising/` and `basilisk_reference_gate/nn/` layer unless a better local boundary already exists.
3. Preserve accepted reference evidence. Do not delete old outputs blindly; archive or isolate them only after preserving accepted raw files.
4. Do not claim either oracle is deployable.
5. Do not include central/native in any NN oracle selector.
6. Implement `hk_cd` with an independent manual central-difference function. Do not compute `hk_cd` by calling `distance_curvature()`.

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
   - compute `hk_cd` using an independent manual central-difference helper, not by calling `distance_curvature()`
   - for smoke runs, record all eligible interface-band samples
   - for full runs, record all samples if eligible count is `<= 50000`; otherwise use deterministic stride or deterministic reservoir sampling capped at `50000` per output time
   - record `eligible_count` and `written_count`
   - ensure all hk values are output on the same cell mask for each time

3. Implement `CD-vs-REF non-degeneracy gate`:
   - compute `e_cd = hk_cd - hk_ref`
   - report `max_abs`, `mean_abs`, `rms`, `p99_abs`, `nonzero_count`
   - report conclusions under thresholds `1e-12`, `1e-10`, `1e-8`
   - if `max_abs <= 1e-12`, declare CD reference-equivalent and exclude it from oracle candidates
   - if CD unexpectedly differs, stop and explain why before running oracle conclusions

4. Implement `ORACLE_NATIVE_MIMIC` only as a local native-mimic diagnostic:
   - `hk_mimic = NN_RAW` if `|hk_raw - hk_ref| <= |hk_d4 - hk_ref|`
   - otherwise `hk_mimic = NN_D4`
   - central/native must remain baseline only
   - report it as "closest to native", not "closest to truth"

5. Implement `ORACLE_DYNAMIC_BEST` as a trajectory-level hindsight selector:
   - compute external score `J_m` for `REF_BG2`, `NN_RAW`, and `NN_D4`
   - use MooNMD/Hysing quantities, not `REF_BG2`, as the benchmark target
   - choose `ORACLE_DYNAMIC_BEST = argmin over {NN_RAW, NN_D4} of J_m`
   - report weights, scales, `J_REF`, `epsilon_effect`, and whether improvement clears the effect-size gate

6. Produce clean evidence package:
   - trajectory comparison table for `REF_BG2`, `NN_RAW`, `NN_D4`, `ORACLE_DYNAMIC_BEST`
   - external-metric table with `E_vc`, `E_yc`, `E_circ`, `E_mass`, and score `J`
   - curvature-probe table for `hk_cd`, `hk_nn_raw`, `hk_nn_d4`, `hk_native_mimic` errors against `hk_ref`
   - figures for `yc(t)`, `vc(t)`, `circ(t)`, and curvature-error distributions
   - external score figure for `J`
   - final Markdown report that states exactly which gates passed or failed

7. Cleanup/archival:
   - keep accepted raw CSVs, status files, compile logs, and probe CSVs
   - archive or isolate old imax/PCA/redistance/sweep figures from the main final report
   - remove generated scratch or `.qcc*` directories only if doing so does not destroy accepted evidence

Stop and report if:

- parity with `REF_BG2` fails
- any method has NaN/Inf before `t=3`
- `CD-vs-REF` shows unexpected nonzero mismatch
- oracle implementation accidentally includes central/native as a selectable candidate
- an apparent improvement does not exceed the effect-size gate

Final response must include:

- exact files changed or created
- exact commands run
- gate results
- whether the outcome is Case A, B, C, or D from the design doc
- a short statement that both oracle concepts are non-deployable diagnostics
```

## 10. Self-Review

- No placeholder requirements remain.
- `central/native` is excluded from the oracle selector when reference-equivalent.
- `ORACLE_NATIVE_MIMIC` is explicitly limited to native-like local curvature information.
- `ORACLE_DYNAMIC_BEST` is explicitly tied to external MooNMD/Hysing trajectory score.
- Both oracle concepts are labeled non-deployable in every relevant section.
- Improvement over native requires an external-metric score and an effect-size gate.
- Cleanup preserves provenance rather than deleting evidence blindly.
- Scope is one implementation plan: probe, native-mimic diagnostic, dynamic-best score, clean report, and archive/isolate old outputs.
