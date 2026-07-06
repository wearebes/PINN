/**
Implementation for ellipse_geometry_clean.h. Compiled separately (plain cc,
not qcc) and linked into the Basilisk host as an object, exactly like
nn/nn_forward_clean.c -- see that header for why.
*/

#include <math.h>
#include <stdbool.h>

#define sq(x) ((x)*(x))
#define sign(x) ((x)>0?1:((x)<0?-1:0))

#include "distance_point_ellipse.h"
#include "ellipse_geometry_clean.h"

double ellipse_geometry_signed_distance (double px, double py, double * qx, double * qy) {
  return DistancePointEllipse (ELLIPSE_A, ELLIPSE_B, px, py, qx, qy);
}

/* Separate entry points (not ellipse_geometry_signed_distance) for callers
   that only need the foot point, not a value destined for the d[] scalar
   field -- kept distinct so Basilisk's dimensional checker does not unify
   this call's outputs with d[]'s length dimension. */
double ellipse_geometry_foot_x (double px, double py) {
  double qx, qy;
  DistancePointEllipse (ELLIPSE_A, ELLIPSE_B, px, py, &qx, &qy);
  return qx;
}

double ellipse_geometry_foot_y (double px, double py) {
  double qx, qy;
  DistancePointEllipse (ELLIPSE_A, ELLIPSE_B, px, py, &qx, &qy);
  return qy;
}

double ellipse_geometry_kappa_analytic (double qx, double qy) {
  double theta = atan2 (qy / ELLIPSE_B, qx / ELLIPSE_A);
  double denom = pow (sq(ELLIPSE_A) * sq(sin(theta)) + sq(ELLIPSE_B) * sq(cos(theta)), 1.5);
  return (ELLIPSE_A * ELLIPSE_B) / denom;
}

double ellipse_geometry_theta (double qx, double qy) {
  return atan2 (qy / ELLIPSE_B, qx / ELLIPSE_A);
}
