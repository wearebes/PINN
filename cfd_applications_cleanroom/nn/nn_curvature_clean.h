#pragma once

#include "nn_features_clean.h"
#include "nn_forward_clean.h"

/* Curvature deployment contract.
 *
 * output_kappa=network_output/Delta
 * D4=8 transforms, averaged in h*kappa space before division by Delta
 *
 * Storage contract for solver hosts:
 *   force_kappa_storage=kappa_nn[]
 *   probe_kappa_storage=kappa_nn_probe[]
 *
 * Mode rules:
 *   NN_DISABLE writes neither storage path.
 *   NN_PROBE_ONLY writes probe_kappa_storage only; force path reads native curvature.
 *   NN27_RAW and NN27_D4 may write force_kappa_storage.
 *   HK_BIAS_PROBE writes CSV diagnostics only and cannot be active with NN_DISABLE or NN_PROBE_ONLY.
 */

static inline double nn_hkappa_d4_clean (const double raw[27]) {
  double sum = 0.0;
  for (int m = 0; m < 8; ++m) {
    double transformed[27];
    nn_d4_transform_one_clean (raw, m, transformed);
    sum += mlp_hkappa_clean (transformed);
  }
  return sum / 8.0;
}

static inline double nn_kappa_raw_clean (const double raw[27], double Delta) {
  return mlp_hkappa_clean (raw) / Delta;
}

static inline double nn_kappa_d4_clean (const double raw[27], double Delta) {
  return nn_hkappa_d4_clean (raw) / Delta;
}
