/**
Constant-kappa oracle host (diagnostic only, lives in tem/).

Same CLSVOF-LS stationary-bubble setup as the cleanroom canary hosts, but the
surface-tension force curvature in the band |d| <= 2*Delta is a compile-time
constant ORACLE_KAPPA instead of an NN prediction. Outside the band the native
distance_curvature() is used, exactly like the NN hosts. This removes ALL
spatial structure from the inserted curvature; comparing Ca against the native
floor and against NN27_RAW isolates whether the NN excess Ca comes from the
spatial structure of its kappa error.
*/

#define JACOBI 1
#ifndef BGHOSTS
# define BGHOSTS 2
#endif

#include "grid/multigrid.h"
#include "navier-stokes/centered.h"
#include "two-phase-clsvof.h"

#include <math.h>
#include <stdlib.h>

#ifndef ORACLE_KAPPA
# error "ORACLE_KAPPA must be defined at compile time"
#endif
#ifndef METHOD_ID
# define METHOD_ID "ORACLE_CONST"
#endif

scalar fn[];
scalar kappa_nn[];

static inline double distance_curvature (Point point, scalar field);

#ifdef ORACLE_RADIAL
/* Parallel-curve radial profile reconstructed from a constant interface
   curvature: kappa(d) = k_i/(1 + d*k_i). For the circle (d = r - R exactly)
   this is 1/r — the exact level-set curvature field. */
#define CLEANROOM_PREFILL_FORCE_KAPPA(field) do { \
  foreach() \
    if (fabs(field[]) <= 2.*Delta) \
      kappa_nn[] = ORACLE_KAPPA/(1. + field[]*ORACLE_KAPPA); \
    else \
      kappa_nn[] = distance_curvature (point, field); \
} while (0)
#else
#define CLEANROOM_PREFILL_FORCE_KAPPA(field) do { \
  foreach() \
    if (fabs(field[]) <= 2.*Delta) \
      kappa_nn[] = ORACLE_KAPPA; \
    else \
      kappa_nn[] = distance_curvature (point, field); \
} while (0)
#endif
#define CLEANROOM_FORCE_KAPPA_VALUE kappa_nn[]
#include "generated/integral_nn_clean.h"

#ifndef LEVEL
# define LEVEL 6
#endif
#ifndef LAPLACE
# define LAPLACE 12000.
#endif
#ifndef CASE_ID
# define CASE_ID "kappa_const_oracle"
#endif

#define DIAMETER 0.8
#define RADIUS (DIAMETER/2.)
#define MU sqrt(DIAMETER/LAPLACE)
#define TMAX (sq(DIAMETER)/MU)

FILE * fp = NULL;
bool have_fn = false;

static const char * cleanroom_env (const char * name, const char * fallback) {
  const char * value = getenv (name);
  return value && value[0] ? value : fallback;
}

static double cleanroom_kappa_linf (void) {
  double kmax = 0.;
  foreach(reduction(max:kmax))
    if (fabs(d[]) <= 2.*Delta) {
      double k = distance_curvature (point, d);
      if (isfinite (k) && fabs(k) > kmax)
        kmax = fabs(k);
    }
  return kmax;
}

static double cleanroom_change_f (void) {
  if (!have_fn) {
    foreach()
      fn[] = f[];
    have_fn = true;
    return 0.;
  }
  return change (f, fn);
}

int main() {
  DT = HUGE [0];
  TOLERANCE = 1e-6 [*];
  stokes = true;
  rho1 = rho2 = 1.;
  mu1 = mu2 = MU;
  const scalar sigma[] = 1.;
  d.sigmaf = sigma;
  N = 1 << LEVEL;
  run();
}

event init (t = 0) {
  if (pid() == 0) {
    fp = fopen ("stationary_trace.csv", "w");
    fprintf (fp, "run_id,repeat_id,benchmark,case_id,method,deployable,evidence_level,level,grid_n,i,t,dt,primary_metric_name,primary_metric_value,tau,Umax,Ca,mass,kappa_linf,kappa_probe_linf,dc\n");
  }
  foreach() {
    d[] = sqrt (sq(x) + sq(y)) - RADIUS;
    kappa_nn[] = nodata;
  }
}

event logfile (i++; t <= TMAX) {
  double dc = cleanroom_change_f();
  scalar un[];
  foreach()
    un[] = norm(u);
  double umax = normf(un).max;
  double Ca = MU*umax;
  double tau = MU*t/sq(DIAMETER);
  double mass = statsf(f).sum;
  double kappa_linf = cleanroom_kappa_linf();
  if (fp) {
    const char * run_id = cleanroom_env ("CLEANROOM_RUN_ID", "unset-run-id");
    const char * repeat_id = cleanroom_env ("CLEANROOM_REPEAT_ID", "0");
    fprintf (fp, "%s,%s,stationary,%s,%s,false,controlled_diagnostic,%d,%d,%d,%.17g,%.17g,Ca,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,nan,%.17g\n",
             run_id, repeat_id, CASE_ID, METHOD_ID, LEVEL, N, i, t, dt,
             Ca, tau, umax, Ca, mass, kappa_linf, dc);
    fflush (fp);
  }
}

event done (t = end) {
  if (fp)
    fclose (fp);
}
