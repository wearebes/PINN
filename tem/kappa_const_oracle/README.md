# Constant-kappa oracle diagnostic (stationary bubble, CLSVOF-LS + integral overlay)

Question: is the NN's +33% excess Ca (vs CLSVOF-LS native, canary
`stationary_canary_20260704T123245Z`: NN27_RAW ratio 1.3314, deterministic) carried by the
**spatial structure** of the kappa error (angular/radial modes), rather than by kappa magnitude
or local roughness? Roughness was already falsified twice (native inserts the roughest field yet
has the lowest Ca; relax filter cut inserted-field roughness 35-43% with Ca response -1.1%/+0.05%).

Method: insert a **constant** kappa over the force band |d|<=2*Delta (native distance_curvature
outside, same band logic as the NN hosts), i.e. a field with zero spatial structure. Compare
same-host Ca_tail_max at L6, full viscous time TMAX = D^2/mu = 78.3837.

Arms (all L6, single run each — canary repeats showed spread 0.0000):

| arm | inserted kappa in band | isolates |
|---|---|---|
| A `native` | stock integral.h path (no overlay) | host floor |
| B `analytic` | 2.5 = 1/R exactly | assembly response to structureless, exact kappa |
| C `nnmean` | 2.492827255493805 (formal L6 interface-band mean of NN27_RAW kappa, from Experiment/report/stationary bubble/summary.csv, run stationary_curvature_process_20260702T121230Z final snapshot) | B + NN mean bias (-0.29%) |

Reference points (same host, from canary/parity evidence extracted 2026-07-05 before results
pruning): native Ca_tail_max ≈ 2.5005e-5; NN27_RAW = 3.3291e-5 (R=1.3314).

## Gates (fixed before running)

- G0 toolchain/floor: Ca_tail_max(A) in [1.25e-5, 5.0e-5] (x2 window around 2.50e-5). Fail -> stop,
  toolchain/flags not comparable, no interpretation.
- G1 completion: every arm reaches t >= 0.999*TMAX with all Ca finite. Fail -> discard arm.
- Readout: R_arm = Ca_tail_max(arm) / Ca_tail_max(A). Tail = max over last 20% of trace rows
  (same definition as the route runner).

## Pre-registered interpretation

- R_B <= 1.05 and R_C <= 1.05  -> CONFIRMED: NN's +33% is carried entirely by spatial error
  structure; mean bias irrelevant; ceiling of any kappa-side fix = native floor; the correct
  lever is source-side systematic error (capwave/training arc), not filtering.
- R_C - R_B > 0.10             -> mean bias matters (unexpected: pressure fails to absorb a
  constant shift in this assembly) — report as its own finding.
- R_B >= 1.25                  -> constant oracle invalid: the assembly needs the radial
  kappa(d)=1/(R+d) band profile; conclusion becomes "radial band structure is load-bearing";
  contingency ARM D (native kappa x scalar ratio) before any further claim.
- Between 1.05 and 1.25        -> partial; report the numeric split, no binary claim.

Non-goals: this is a diagnostic oracle, NOT a deployable method (a constant cannot generalize
beyond the circle). No paper-facing claim; evidence level = controlled diagnostic in tem/.
