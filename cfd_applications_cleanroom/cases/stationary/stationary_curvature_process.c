/**
Cleanroom stationary-bubble process curvature diagnostic.

This host advances the same CLSVOF-LS neural-network surface-tension path used
by the deployable stationary canary and exports native numeric curvature versus
NN-predicted curvature at fixed fractions of the total stationary-bubble time.
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
#define NN27_D4_MODE 3
#define NN27_RAW_RELAX_MODE 4
#define NN27_D4_RELAX_MODE 5
#define CLEANROOM_RELAX_ENABLED \
  (NN_MODE == NN27_RAW_RELAX_MODE || NN_MODE == NN27_D4_RELAX_MODE)

#ifndef NN_MODE
# define NN_MODE NN27_RAW_MODE
#endif
#ifndef NN_WEIGHTS_HEADER
# define NN_WEIGHTS_HEADER "generated/nn_weights_nn27_r128_clean.h"
#endif

#include "nn_forward_clean.h"
#include "nn_curvature_clean.h"

#define CLEANROOM_USE_NN_FORCE_CURVATURE 1

scalar fn[];
scalar kappa_nn[];

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

static inline double cleanroom_method_hkappa (const double raw[27]) {
#if NN_MODE == NN27_D4_MODE || NN_MODE == NN27_D4_RELAX_MODE
  return nn_hkappa_d4_clean (raw);
#else
  return mlp_hkappa_clean (raw);
#endif
}

#include "stationary_relax_filter.h"

static inline double cleanroom_nn_force_curvature (Point point, scalar field) {
  double raw[27];
  cleanroom_build_raw_from_d (point, field, raw);
  double hk = cleanroom_method_hkappa (raw);
  return hk/Delta;
}

#if CLEANROOM_RELAX_ENABLED
#define CLEANROOM_PREFILL_FORCE_KAPPA(field) do { \
  scalar hk_raw_tmp[]; \
  CLEANROOM_FILL_HK_RAW_BAND (field, hk_raw_tmp); \
  foreach() { \
    if (fabs(field[]) <= 2.*Delta) { \
      CleanroomHkappaSample sample = cleanroom_relax_sample (point, field, hk_raw_tmp); \
      kappa_nn[] = sample.hk_force/Delta; \
    } \
    else \
      kappa_nn[] = distance_curvature (point, field); \
  } \
} while (0)
#else
#define CLEANROOM_PREFILL_FORCE_KAPPA(field) do { \
  foreach() \
    if (fabs(field[]) <= 2.*Delta) \
      kappa_nn[] = cleanroom_nn_force_curvature (point, field); \
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
# define CASE_ID "stationary_curvature_process"
#endif
#ifndef METHOD_ID
# if NN_MODE == NN27_D4_MODE
#  define METHOD_ID "NN27_D4"
# elif NN_MODE == NN27_RAW_RELAX_MODE
#  define METHOD_ID "NN27_RAW_RELAX"
# elif NN_MODE == NN27_D4_RELAX_MODE
#  define METHOD_ID "NN27_D4_RELAX"
# else
#  define METHOD_ID "NN27_RAW"
# endif
#endif

#define DIAMETER 0.8
#define RADIUS (DIAMETER/2.)
#define MU sqrt(DIAMETER/LAPLACE)
#define SIGMA 1.
#define TMAX (sq(DIAMETER)/MU)

FILE * fp_field = NULL;
FILE * fp_trace = NULL;
bool have_fn = false;
int next_snapshot = 0;
static const double snapshot_targets[4] = {0., 1./3., 2./3., 1.};

static const char * cleanroom_env (const char * name, const char * fallback) {
  const char * value = getenv (name);
  return value && value[0] ? value : fallback;
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

static double cleanroom_umax (void) {
  scalar un[];
  foreach()
    un[] = norm(u);
  return normf(un).max;
}

static void cleanroom_write_snapshot (int iter, int snapshot_index, double snapshot_fraction) {
  const char * run_id = cleanroom_env ("CLEANROOM_RUN_ID", "unset-run-id");
  double tau = MU*t/sq(DIAMETER);
  double umax = cleanroom_umax();
  double Ca = MU*umax/SIGMA;
  scalar hk_raw_tmp[];
  CLEANROOM_FILL_HK_RAW_BAND (d, hk_raw_tmp);
  foreach()
    if (fabs(d[]) <= 2.*Delta) {
      double native_kappa = distance_curvature (point, d);
      double hk_native = Delta*native_kappa;
      double hk_nn_raw = hk_raw_tmp[];
      double hk_nn_force = hk_nn_raw;
      int relax_neighbor_count = 0;
      double relax_lambda = 0.;
#if CLEANROOM_RELAX_ENABLED
      CleanroomHkappaSample sample = cleanroom_relax_sample (point, d, hk_raw_tmp);
      hk_nn_force = sample.hk_force;
      relax_neighbor_count = sample.relax_neighbor_count;
      relax_lambda = CLEANROOM_RELAX_LAMBDA;
#endif
      double hk_nn = hk_nn_force;
      double kappa_model = hk_nn/Delta;
      double delta_hk_raw = hk_nn_raw - hk_native;
      double delta_hk_force = hk_nn_force - hk_native;
      double kappa_nn_from_hk_over_delta = hk_nn/Delta;
      double scale_identity_error = kappa_model - kappa_nn_from_hk_over_delta;
      double sign_product = kappa_model*native_kappa;
      double theta_rad = atan2 (y, x);
      double theta_deg = theta_rad*180./pi;
      if (theta_deg < 0.)
        theta_deg += 360.;
      if (fp_field)
        fprintf (fp_field,
          "%s,stationary,%s,%s,%d,%d,%d,%.17g,%d,%.17g,%.17g,%.17g,%.17g,"
          "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
          "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d\n",
          run_id, CASE_ID, METHOD_ID, LEVEL, N, iter, t, snapshot_index, snapshot_fraction, tau, SIGMA,
          x, y, theta_deg, d[], fabs(d[])/Delta, hk_native, hk_nn, hk_nn - hk_native,
          native_kappa, kappa_model, kappa_model - native_kappa, kappa_nn_from_hk_over_delta,
          scale_identity_error, sign_product, Ca, hk_nn_raw, hk_nn_force, delta_hk_raw,
          delta_hk_force, relax_lambda, relax_neighbor_count);
    }
}

int main() {
  DT = HUGE [0];
  TOLERANCE = 1e-6 [*];
  stokes = true;
  rho1 = rho2 = 1.;
  mu1 = mu2 = MU;
  const scalar sigma[] = SIGMA;
  d.sigmaf = sigma;
  N = 1 << LEVEL;
  run();
}

event init (t = 0) {
  if (pid() == 0) {
    fp_field = fopen ("curvature_process.csv", "w");
    fprintf (fp_field,
      "run_id,benchmark,case_id,method,level,grid_n,i,t,snapshot_index,snapshot_fraction,tau,"
      "sigma,x,y,theta_deg,d,abs_d_over_delta,hk_native,hk_nn,delta_hk,"
      "native_kappa,kappa_nn,delta_kappa,kappa_nn_from_hk_over_delta,scale_identity_error,"
      "sign_product,Ca,hk_nn_raw,hk_nn_force,delta_hk_raw,delta_hk_force,relax_lambda,"
      "relax_neighbor_count\n");
    fp_trace = fopen ("surface_tension_trace.csv", "w");
    fprintf (fp_trace,
      "run_id,repeat_id,benchmark,case_id,method,deployable,evidence_level,level,grid_n,i,t,dt,"
      "primary_metric_name,primary_metric_value,tau,sigma,mu,Umax,Ca,mass,kappa_linf,dc\n");
  }
  foreach() {
    d[] = sqrt (sq(x) + sq(y)) - RADIUS;
    kappa_nn[] = nodata;
  }
}

event process_log (i++; t <= TMAX) {
  double dc = cleanroom_change_f();
  double umax = cleanroom_umax();
  double Ca = MU*umax/SIGMA;
  double tau = MU*t/sq(DIAMETER);
  double mass = statsf(f).sum;
  double kappa_linf = cleanroom_kappa_linf();
  if (fp_trace) {
    const char * run_id = cleanroom_env ("CLEANROOM_RUN_ID", "unset-run-id");
    const char * repeat_id = cleanroom_env ("CLEANROOM_REPEAT_ID", "0");
    fprintf (fp_trace,
      "%s,%s,stationary,%s,%s,true,controlled_diagnostic,%d,%d,%d,%.17g,%.17g,"
      "Ca,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
      run_id, repeat_id, CASE_ID, METHOD_ID, LEVEL, N, i, t, dt, Ca, tau, SIGMA, MU, umax, Ca,
      mass, kappa_linf, dc);
    fflush (fp_trace);
  }
}

event process_snapshots (i++; t <= TMAX) {
  double tau = MU*t/sq(DIAMETER);
  while (next_snapshot < 4 && tau + 1e-12 >= snapshot_targets[next_snapshot]) {
    cleanroom_write_snapshot (i, next_snapshot, snapshot_targets[next_snapshot]);
    next_snapshot++;
  }
  if (fp_field)
    fflush (fp_field);
}

event done (t = end) {
  while (next_snapshot < 4) {
    cleanroom_write_snapshot (i, next_snapshot, snapshot_targets[next_snapshot]);
    next_snapshot++;
  }
  if (fp_field)
    fclose (fp_field);
  if (fp_trace)
    fclose (fp_trace);
}
