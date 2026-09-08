#pragma once

#include "nn_features_clean.h"
#include "nn_forward_clean.h"

/* Curvature deployment contract.
 *
 * output_kappa=network_output/Delta
 * Storage contract for solver hosts:
 *   force_kappa_storage=kappa_nn[]
 *
 * Mode rules:
 *   NN27_RAW writes force_kappa_storage.
 */

static inline double nn_kappa_raw_clean (const double raw[27], double Delta) {
  return mlp_hkappa_clean (raw) / Delta;
}
