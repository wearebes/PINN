#pragma once

/* Clean NN forward ABI.
 *
 * Contract:
 *   input_dim=27
 *   raw27=[phi9/h, nx9, ny9]
 *   target=h*kappa
 */

#ifndef RAW_IN
# define RAW_IN 27
#endif

#ifndef MLP_IN
# define MLP_IN 27
#endif

#ifndef MLP_H
# define MLP_H 128
#endif

void nn_prepare_input_clean (const double raw[RAW_IN], double z[MLP_IN]);
double mlp_hkappa_clean (const double raw[RAW_IN]);
