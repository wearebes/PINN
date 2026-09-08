/**
Cleanroom stationary-bubble initial-field curvature diagnostic.

This host does not advance the flow. It initializes the same quadrant circle
level-set field used by the stationary CLSVOF canary and exports native vs NN
curvature values on the force/interface band.
*/

#define JACOBI 1
#ifndef BGHOSTS
# define BGHOSTS 2
#endif

#include "grid/multigrid.h"
#include "navier-stokes/centered.h"
#include "two-phase-clsvof.h"
#include "integral.h"

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

#ifndef LEVEL
# define LEVEL 6
#endif
#ifndef CASE_ID
# define CASE_ID "stationary_curvature_field"
#endif
#ifndef METHOD_ID
# define METHOD_ID "NN27_RAW"
#endif

#define DIAMETER 0.8
#define RADIUS (DIAMETER/2.)

FILE * fp = NULL;

static const char * cleanroom_env (const char * name, const char * fallback) {
  const char * value = getenv (name);
  return value && value[0] ? value : fallback;
}

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
  return mlp_hkappa_clean (raw);
}

int main() {
  DT = HUGE [0];
  TOLERANCE = 1e-6 [*];
  stokes = true;
  rho1 = rho2 = 1.;
  mu1 = mu2 = 1.;
  const scalar sigma[] = 1.;
  d.sigmaf = sigma;
  N = 1 << LEVEL;
  run();
}

event init (i = 0) {
  foreach()
    d[] = sqrt (sq(x) + sq(y)) - RADIUS;
}

event curvature_export (i = 0) {
  if (pid() == 0) {
    fp = fopen ("curvature_field.csv", "w");
    fprintf (fp,
      "run_id,benchmark,case_id,method,level,grid_n,x,y,theta,d,abs_d_over_delta,"
      "hk_native,hk_nn,delta_hk,native_kappa,kappa_nn,delta_kappa,"
      "kappa_nn_from_hk_over_delta,scale_identity_error,sign_product,"
      "hk_nn_raw,hk_nn_force,delta_hk_raw,delta_hk_force\n");
  }
  const char * run_id = cleanroom_env ("CLEANROOM_RUN_ID", "unset-run-id");
  foreach()
    if (fabs(d[]) <= 2.*Delta) {
      double native_kappa = distance_curvature (point, d);
      double hk_native = Delta*native_kappa;
      double raw[27];
      cleanroom_build_raw_from_d (point, d, raw);
      double hk_nn_raw = cleanroom_method_hkappa (raw);
      double hk_nn_force = hk_nn_raw;
      double hk_nn = hk_nn_force;
      double kappa_nn = hk_nn/Delta;
      double delta_hk = hk_nn - hk_native;
      double delta_hk_raw = hk_nn_raw - hk_native;
      double delta_hk_force = hk_nn_force - hk_native;
      double delta_kappa = kappa_nn - native_kappa;
      double kappa_nn_from_hk_over_delta = hk_nn/Delta;
      double scale_identity_error = kappa_nn - kappa_nn_from_hk_over_delta;
      double sign_product = kappa_nn*native_kappa;
      double theta = atan2 (y, x);
      if (fp)
        fprintf (fp,
          "%s,stationary,%s,%s,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,"
          "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
          "%.17g,%.17g,%.17g,%.17g\n",
          run_id, CASE_ID, METHOD_ID, LEVEL, N, x, y, theta, d[], fabs(d[])/Delta,
          hk_native, hk_nn, delta_hk, native_kappa, kappa_nn, delta_kappa,
          kappa_nn_from_hk_over_delta, scale_identity_error, sign_product,
          hk_nn_raw, hk_nn_force, delta_hk_raw, delta_hk_force);
    }
  if (fp)
    fclose (fp);
  return 1;
}
