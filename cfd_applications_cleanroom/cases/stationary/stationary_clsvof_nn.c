/**
Cleanroom stationary-bubble CLSVOF-LS neural-network overlay host.
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

#define NN27_RAW_MODE 2

#ifndef NN_MODE
# define NN_MODE NN27_RAW_MODE
#endif
#ifndef NN_WEIGHTS_HEADER
# define NN_WEIGHTS_HEADER "generated/nn_weights_nn27_r128_clean.h"
#endif

#include "nn_forward_clean.h"
#include "nn_curvature_clean.h"

#define DIAMETER 0.8
#define RADIUS (DIAMETER/2.)

#if NN_MODE == NN27_RAW_MODE
# define CLEANROOM_USE_NN_FORCE_CURVATURE 1
#else
# error "stationary_clsvof_nn.c only supports NN_MODE=NN27_RAW_MODE"
#endif

scalar fn[];
scalar kappa_nn[];

#if CLEANROOM_USE_NN_FORCE_CURVATURE
static inline double distance_curvature (Point point, scalar field);

static inline void cleanroom_build_raw_from_d (Point point, scalar field, double raw[27]) {
  double D[5][5];
  D[0][0] = field[-2,-2]; D[0][1] = field[-2,-1]; D[0][2] = field[-2,0]; D[0][3] = field[-2,1]; D[0][4] = field[-2,2];
  D[1][0] = field[-1,-2]; D[1][1] = field[-1,-1]; D[1][2] = field[-1,0]; D[1][3] = field[-1,1]; D[1][4] = field[-1,2];
  D[2][0] = field[0,-2];  D[2][1] = field[0,-1];  D[2][2] = field[0,0];  D[2][3] = field[0,1];  D[2][4] = field[0,2];
  D[3][0] = field[1,-2];  D[3][1] = field[1,-1];  D[3][2] = field[1,0];  D[3][3] = field[1,1];  D[3][4] = field[1,2];
  D[4][0] = field[2,-2];  D[4][1] = field[2,-1];  D[4][2] = field[2,0];  D[4][3] = field[2,1];  D[4][4] = field[2,2];
  nn_build_raw27_clean (D, Delta, 1.0, raw);
}

static inline double cleanroom_nn_force_curvature (Point point, scalar field) {
  double raw[27];
  cleanroom_build_raw_from_d (point, field, raw);
  double hk = mlp_hkappa_clean (raw);
  double native_k = distance_curvature (point, field);
  double denom = Delta*native_k;
  double ratio = fabs(denom) > 1e-300 ? hk/denom : 0.;
  return native_k*ratio;
}

# define CLEANROOM_PREFILL_FORCE_KAPPA(field) do { \
  foreach() \
    if (fabs(field[]) <= 2.*Delta) \
      kappa_nn[] = cleanroom_nn_force_curvature (point, field); \
    else \
      kappa_nn[] = distance_curvature (point, field); \
} while (0)
# define CLEANROOM_FORCE_KAPPA_VALUE kappa_nn[]
#include "generated/integral_nn_clean.h"
#else
#include "integral.h"
#endif

#ifndef LEVEL
# define LEVEL 6
#endif
#ifndef LAPLACE
# define LAPLACE 12000.
#endif
#ifndef CASE_ID
# define CASE_ID "stationary_clsvof_nn"
#endif
#ifndef METHOD_ID
# if NN_MODE == NN27_RAW_MODE
#  define METHOD_ID "NN27_RAW"
# endif
#endif

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
  double kappa_probe_linf = nodata;
  if (fp) {
    const char * run_id = cleanroom_env ("CLEANROOM_RUN_ID", "unset-run-id");
    const char * repeat_id = cleanroom_env ("CLEANROOM_REPEAT_ID", "0");
    fprintf (fp, "%s,%s,stationary,%s,%s,false,runtime_smoke,%d,%d,%d,%.17g,%.17g,Ca,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
             run_id, repeat_id, CASE_ID, METHOD_ID, LEVEL, N, i, t, dt,
             Ca, tau, umax, Ca, mass, kappa_linf, kappa_probe_linf, dc);
    fflush (fp);
  }
}

event done (t = end) {
  if (fp)
    fclose (fp);
}
