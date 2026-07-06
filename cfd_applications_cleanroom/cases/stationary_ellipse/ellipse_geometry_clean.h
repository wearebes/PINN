/**
Route-local ellipse geometry for the stationary_ellipse diagnostic family.

Declarations only. The implementation lives in ellipse_geometry_clean.c and
is compiled to a separate object and linked in, exactly like
nn/nn_forward_clean.c -- this keeps Basilisk's dimensional checker from
tracing through the arithmetic (ELLIPSE_A/ELLIPSE_B are plain numeric
literals with no physical-length dimension annotation, and qcc's checker
mis-infers a length dimension for hk_analytic if it can inline the body;
see docs/superpowers/plans/2026-07-03-stationary-ellipse-curvature-stress-diagnostic.md
section 7 for the empirical diagnosis). No new projection algorithm is
implemented here -- ellipse_geometry_clean.c wraps the official vendor
`distance_point_ellipse.h` (Eberly's exact point-to-ellipse signed distance).
*/

#ifndef ELLIPSE_A
# define ELLIPSE_A 0.4472136
#endif
#ifndef ELLIPSE_B
# define ELLIPSE_B 0.3577709
#endif

double ellipse_geometry_signed_distance (double px, double py, double * qx, double * qy);
double ellipse_geometry_foot_x (double px, double py);
double ellipse_geometry_foot_y (double px, double py);
double ellipse_geometry_kappa_analytic (double qx, double qy);
double ellipse_geometry_theta (double qx, double qy);
