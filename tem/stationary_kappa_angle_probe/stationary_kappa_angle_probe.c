/*
 * Temporary stationary-bubble kappa-vs-angle diagnostic.
 *
 * Runs the N=64 CLSVOF-LS native-force path and exports native and NN
 * curvature estimates on the same level-set band at selected iterations.
 */

#define JACOBI 1
#define BGHOSTS 2

#include "grid/multigrid.h"
#include "navier-stokes/centered.h"
#include "two-phase-clsvof.h"

/* Compile with -DNN_DISABLE=1 so the force path stays native CLSVOF-LS. */
#include "integral_nn.h"

#ifndef LEVEL
# define LEVEL 6
#endif
#ifndef STOP_ITER
# define STOP_ITER 15000
#endif

#define DIAMETER 0.8
#define RADIUS (DIAMETER/2.)
#define SIGMA 1.0
#define RHO 1.0
#define LAPLACE 12000.
#define MU sqrt(SIGMA*DIAMETER/(RHO*LAPLACE))

static FILE * kfp = NULL;

static void export_kappa_band (int iter, double time)
{
  if (pid() == 0 && !kfp) {
    kfp = fopen ("stationary_kappa_angle_probe.csv", "w");
    fprintf (kfp,
      "base_force,level,N,i,t,tau,x,y,theta_deg_quadrant,abs_d_over_delta,"
      "native_kappa,nn_raw_kappa,nn_d4_kappa,delta_raw_minus_native,"
      "delta_d4_minus_native,exact_signed_kappa\n");
  }

  foreach()
    if (fabs(d[]) <= 2.*Delta) {
      double native_k = distance_curvature (point, d);
      double raw_k = nn_raw_kappa (point, d, (double) NN_SGN);
      double d4_k = nn_d4_kappa (point, d, (double) NN_SGN);
      double theta = atan2 (y, x)*180./pi;
      if (theta < 0.)
        theta += 360.;
      if (kfp)
        fprintf (kfp,
          "CLSVOF_LS_NATIVE,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
          "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
          LEVEL, N, iter, time, MU*time/sq(DIAMETER), x, y, theta,
          fabs(d[])/Delta, native_k, raw_k, d4_k, raw_k - native_k,
          d4_k - native_k, -1./RADIUS);
    }
  if (kfp)
    fflush (kfp);
}

int main()
{
  size (1.0);
#if FULL_CIRCLE
  origin (-0.5, -0.5);
#else
  origin (0., 0.);
#endif
  DT = HUGE [0];
  TOLERANCE = 1e-6 [*];
  stokes = true;
  rho1 = rho2 = RHO;
  mu1 = mu2 = MU;
  const scalar sigma_const[] = SIGMA;
  d.sigmaf = sigma_const;
  N = 1 << LEVEL;
  run();
}

event init (i = 0)
{
  foreach()
    d[] = RADIUS - sqrt (sq(x) + sq(y));

  vertex scalar phi[];
  foreach_vertex()
    phi[] = (d[] + d[-1] + d[0,-1] + d[-1,-1])/4.;
  fractions (phi, f);
}

event kappa_probe (i += 5000; i <= STOP_ITER)
{
  export_kappa_band (i, t);
}

event stop_probe (i = STOP_ITER)
{
  if (kfp)
    fclose (kfp);
  return 1;
}
