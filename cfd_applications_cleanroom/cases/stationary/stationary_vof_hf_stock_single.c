/**
Cleanroom stationary-bubble VOF-HF stock reference.

This host reproduces the official Basilisk spurious-current setup for one
LEVEL/LAPLACE pair on a uniform multigrid. It emits the cleanroom common raw
trace schema plus stationary-specific Ca diagnostics. It is a reference gate,
not an NN deployment host.
*/

#define JACOBI 1

#include "grid/multigrid.h"
#include "navier-stokes/centered.h"
#include "vof.h"
#include "tension.h"

#include <stdlib.h>

scalar c[], * interfaces = {c};
scalar cn[];

#ifndef LEVEL
# define LEVEL 5
#endif
#ifndef LAPLACE
# define LAPLACE 12000.
#endif
#ifndef CASE_ID
# define CASE_ID "stationary_vof_hf_stock_single"
#endif

#define DIAMETER 0.8
#define MU sqrt(DIAMETER/LAPLACE)
#define TMAX (sq(DIAMETER)/MU)

FILE * fp = NULL;

static const char * cleanroom_env (const char * name, const char * fallback) {
  const char * value = getenv (name);
  return value && value[0] ? value : fallback;
}

int main() {
  DT = HUGE [0];
  TOLERANCE = 1e-6 [*];
  stokes = true;
  c.sigma = 1.;
  N = 1 << LEVEL;
  run();
}

event init (i = 0) {
  mu[] = {MU, MU};
  if (pid() == 0) {
    fp = fopen ("stationary_trace.csv", "w");
    fprintf (fp, "run_id,repeat_id,benchmark,case_id,method,deployable,evidence_level,level,grid_n,i,t,dt,primary_metric_name,primary_metric_value,tau,Umax,Ca,dc\n");
  }
  fraction (c, sq(DIAMETER/2.) - sq(x) - sq(y));
  foreach()
    cn[] = c[];
}

event logfile (i++; t <= TMAX) {
  double dc = change (c, cn);
  scalar un[];
  foreach()
    un[] = norm(u);
  double umax = normf(un).max;
  double Ca = MU*umax;
  double tau = MU*t/sq(DIAMETER);
  if (fp) {
    const char * run_id = cleanroom_env ("CLEANROOM_RUN_ID", "unset-run-id");
    const char * repeat_id = cleanroom_env ("CLEANROOM_REPEAT_ID", "0");
    fprintf (fp, "%s,%s,stationary,%s,VOF_HF_NATIVE,true,runtime_smoke,%d,%d,%d,%.17g,%.17g,Ca,%.17g,%.17g,%.17g,%.17g,%.17g\n",
             run_id, repeat_id, CASE_ID, LEVEL, N, i, t, dt, Ca, tau, umax, Ca, dc);
    if (i % 4096 == 0)
      fflush (fp);
  }
}

event done (t = end) {
  if (fp)
    fclose (fp);
}
