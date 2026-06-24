/**
# Rising bubble -- native |h*kappa| interface-band distribution probe (plan SS2.4 OOD check)

Scratch diagnostic, NOT part of the tested P2-GATE ladder. Pure native CLSVOF-LS
(identical physics/IC/BC to basilisk_reference_gate/rising/cases/rising_bubble.c),
with one added event that dumps quantiles of |kappa| and |h*kappa| over the
interface band (|d| < 2*Delta) at t in {0,0.5,1,1.5,2,2.5,3}, to stderr.

Question: does the bubble's curvature ever leave the low-eta regime (where
central-difference and the NN models agree closely, per the rising-bubble run)
and reach into the eta > 0.1 regime where the offline circle/ellipse tests show
large NN-vs-central-diff divergence (memory: error is ~99.8% eta-driven there)?
*/

#include "grid/multigrid.h"
#include "navier-stokes/centered.h"
#include "two-phase-clsvof.h"
#include "integral.h"

#ifndef LEVEL
# define LEVEL 8
#endif
#ifndef TEND
# define TEND 3.0
#endif

u.t[right] = dirichlet(0);
u.t[left]  = dirichlet(0);

int main() {
  dimensions (nx = 4);
  size (2 [1]);
  DT = 1. [0,1];
  init_grid (1 << LEVEL);
  rho1 = 1000.[0], mu1 = 10.;
  rho2 = 100.,     mu2 = 1.;
  const scalar sigma[] = 24.5;
  d.sigmaf = sigma;
  TOLERANCE = 1e-4 [*];
  run();
}

event init (t = 0) {
  foreach()
    d[] = sqrt (sq(x - 0.5) + sq(y)) - 0.25;
}

event acceleration (i++) {
  face vector av = a;
  foreach_face(x)
    av.x[] -= 0.98;
}

static int cmp_double (const void * a, const void * b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

event kappa_probe (t = {0, 0.5, 1, 1.5, 2, 2.5, 3}) {
  int n = 0;
  foreach (reduction(+:n))
    if (fabs(d[]) < 2.*Delta)
      n++;

  double * hk = malloc (n*sizeof(double));
  int idx = 0;
  foreach()
    if (fabs(d[]) < 2.*Delta) {
      double kappa = distance_curvature (point, d);
      if (idx < n)
        hk[idx] = fabs(kappa)*Delta;
      idx++;
    }
  int n_check = idx;

  qsort (hk, n, sizeof(double), cmp_double);

  int n_gt01 = 0;
  for (int i = 0; i < n; i++)
    if (hk[i] > 0.1) n_gt01++;

  double probs[7] = {0.001, 0.01, 0.05, 0.50, 0.95, 0.99, 0.999};
  fprintf (stderr, "KAPPA_PROBE t=%.4f n=%d n_check=%d frac_eta_gt_0.1=%.5f hk_min=%.6f hk_max=%.6f",
           t, n, n_check, (double)n_gt01/n, hk[0], hk[n-1]);
  for (int j = 0; j < 7; j++) {
    int k = (int)(probs[j]*(n-1));
    if (k < 0) k = 0;
    if (k >= n) k = n-1;
    fprintf (stderr, " p%g=%.6f", probs[j]*100., hk[k]);
  }
  fprintf (stderr, "\n");

  free (hk);
}

event end (t = TEND) {}
