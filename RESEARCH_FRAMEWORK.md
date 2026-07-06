# PINN Curvature Closure Research Framework

## 1. Scientific Question

This project is a numerical-methods research project, not only a model-training or code-integration project.

The central question is:

> Can neural-network curvature closures improve, preserve, or explain solver behavior when inserted into existing online two-phase-flow solver paths?

The target claim is not that NN curvature is globally better than every curvature method. The target claim must be tied to a specific host solver, benchmark, baseline, closure variant, and evidence artifact.

## 2. Research Object

The learned object is a stencil-to-scalar curvature closure:

- input: local level-set stencil information near the interface, such as `phi9` or `phi9 + nx9 + ny9`
- output: a dimensionless discrete curvature target, typically `h*kappa`
- intended use: replace or augment the curvature value consumed by an existing surface-tension force assembly

The NN closure is not a complete solver. It is a curvature component inside a host solver.

## 3. Host Lines

### 3.1 Basilisk Line

The Basilisk application line evaluates NN curvature closure inside the existing `CLSVOF-LS` host path.

Main comparison:

| Role | Method | Interpretation |
|---|---|---|
| Research baseline | `CLSVOF-LS` native curvature | Same-host baseline |
| Research method | `CLSVOF-LS + NN curvature` | Main object of study |
| Positive control | `VOF-HF` | Well-balanced/reference-quality control, not the replacement research target |

Valid Basilisk conclusion form:

> In the Basilisk `CLSVOF-LS` host, NN curvature closure improves / preserves / worsens behavior relative to the same-host native `CLSVOF-LS` baseline under benchmark X.

Invalid conclusion form:

> `VOF-HF` is very accurate, therefore NN curvature has solved `CLSVOF-LS`.

### 3.2 TwoPhaseFlow/OpenFOAM Line

The TwoPhaseFlow/OpenFOAM line must be evaluated independently inside the TwoPhaseFlow/OpenFOAM host.

Basilisk evidence can guide design choices, diagnostic metrics, and expected failure modes, but it does not prove the OpenFOAM-side result.

Valid OpenFOAM conclusion form:

> In the TwoPhaseFlow/OpenFOAM host, NN curvature closure improves / preserves / worsens behavior relative to the same-host native baseline under benchmark X.

## 4. Closure Contract

Every experiment or report must state the closure contract before interpreting results:

| Field | Required content |
|---|---|
| Host | Basilisk `CLSVOF-LS`, TwoPhaseFlow/OpenFOAM, or another explicit host |
| Benchmark | stationary bubble, rising bubble, capillary wave, flower, ellipse, split test, etc. |
| Baseline | same-host native method |
| Positive control | optional; must not replace the same-host baseline |
| NN variant | V1, V2, V2.2, V2.3, V3, Part 2 DCTS variant, etc. |
| Input feature | e.g. `phi9/h`, `phi9/h + nx9 + ny9`, PCA-18 |
| Target | e.g. `h*kappa`, `alpha*h*kappa`; target must not be confused with input |
| Checkpoint | model path, training data, seed or run identifier when available |
| Normalization | normalization CSV or embedded transform |
| Runtime insertion | where the NN curvature enters the solver force path |
| Metrics | solver-facing metrics and curvature diagnostics |
| Artifact | table, figure, log, report, or reproducible output path |

If these fields are not fixed, the conclusion must be treated as preliminary.

## 5. Evidence Levels

Use explicit evidence levels in all research discussions.

| Level | Meaning | Allowed claim |
|---|---|---|
| Implemented | Code path exists | The method is implemented |
| Smoke-tested | Minimal run exits and produces outputs | The path can run on a small case |
| Diagnosed | Outputs were inspected with targeted metrics or plots | A specific behavior was observed |
| Benchmark-validated | Full benchmark comparison against the correct baseline is available | The method improves / preserves / worsens under that benchmark |
| Paper-ready | Result is reproducible, labeled correctly, and supported by figures/tables | The claim can enter manuscript text |

A clean command exit is not scientific validation.

## 6. Benchmark Roles

Benchmarks must be interpreted by role:

| Benchmark family | Main purpose |
|---|---|
| Split test | In-distribution regression accuracy |
| Ellipse | Controlled OOD geometry and curvature behavior |
| Flower | Strong OOD geometry plus reinitialization robustness |
| Stationary bubble | Static balance and spurious-current behavior |
| Rising bubble | Dynamic solver behavior against benchmark trajectory metrics |
| Capillary wave | Surface-tension dynamics against physical or regression references |

Regression metrics alone are not enough for CFD application claims. Solver-facing behavior must be measured on the host path.

## 7. Claim Rules

1. Same-host comparison is mandatory for application claims.
2. Positive controls must be labeled as positive controls.
3. Basilisk and TwoPhaseFlow/OpenFOAM evidence must not be merged.
4. Curvature-regression success does not automatically imply solver success.
5. Solver success must be tied to benchmark metrics, not only visual plausibility.
6. NN closure must be described as a curvature replacement inside a host operator, not as an all-domain neural solver.
7. A result can be useful even if it does not outperform the strongest positive control, provided it improves, preserves, or explains behavior relative to the correct native baseline.

## 8. Reporting Order

Research reports should use this order:

1. Research question and host line
2. Benchmark identity and baseline roles
3. Closure contract
4. Metrics and figures
5. Result table
6. Evidence-level conclusion
7. Failure modes or limitations
8. Next controlled experiment

Prefer evidence-first writing: tables, figures, and metrics before code walkthroughs.

## 9. Current Paper-Facing Wording

For the Basilisk `CLSVOF-LS` deployment, use:

> interfacial-band neural-curvature replacement inside the existing `CLSVOF-LS` surface-tension operator

Avoid broader wording such as:

- all-domain neural curvature solver
- neural replacement of the whole two-phase solver
- proof that `VOF-HF` behavior transfers to `CLSVOF-LS + NN`

## 10. Working Rule

Experiments may continue while this framework is being refined.

The purpose of this document is to fix the interpretation standard: every new run should be mapped back to host line, baseline role, closure contract, evidence level, and allowed claim.
