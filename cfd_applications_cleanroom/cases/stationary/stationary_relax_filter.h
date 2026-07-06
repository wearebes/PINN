/**
One-step 4-neighbor local relaxation of NN h*kappa (experimental *_RELAX
methods). Single source of truth for the filter formula -- hosts must not
duplicate it. Include after cleanroom_build_raw_from_d() and
cleanroom_method_hkappa() are defined.
*/
#pragma once

#ifndef CLEANROOM_RELAX_LAMBDA
# define CLEANROOM_RELAX_LAMBDA 0.25
#endif
#ifndef CLEANROOM_FORCE_BAND_WIDTH
# define CLEANROOM_FORCE_BAND_WIDTH 2.
#endif

typedef struct {
  double hk_raw;
  double hk_force;
  int relax_neighbor_count;
} CleanroomHkappaSample;

static inline bool cleanroom_relax_neighbor_valid (double hk_value,
                                                   double d_value,
                                                   double delta_value) {
  return hk_value != nodata && isfinite (hk_value) &&
    fabs(d_value) <= CLEANROOM_FORCE_BAND_WIDTH*delta_value;
}

static inline CleanroomHkappaSample
cleanroom_relax_sample (Point point, scalar field, scalar hk_raw) {
  CleanroomHkappaSample sample;
  sample.hk_raw = hk_raw[];
  sample.hk_force = hk_raw[];
  sample.relax_neighbor_count = 0;
  double numerator = hk_raw[];
  double denominator = 1.;
  foreach_dimension() {
    if (cleanroom_relax_neighbor_valid (hk_raw[1], field[1], Delta)) {
      numerator += CLEANROOM_RELAX_LAMBDA*hk_raw[1];
      denominator += CLEANROOM_RELAX_LAMBDA;
      sample.relax_neighbor_count++;
    }
    if (cleanroom_relax_neighbor_valid (hk_raw[-1], field[-1], Delta)) {
      numerator += CLEANROOM_RELAX_LAMBDA*hk_raw[-1];
      denominator += CLEANROOM_RELAX_LAMBDA;
      sample.relax_neighbor_count++;
    }
  }
  sample.hk_force = numerator/denominator;
  return sample;
}

#define CLEANROOM_FILL_HK_RAW_BAND(field, hk_raw) do { \
  foreach() { \
    if (fabs(field[]) <= CLEANROOM_FORCE_BAND_WIDTH*Delta) { \
      double raw[27]; \
      cleanroom_build_raw_from_d (point, field, raw); \
      hk_raw[] = cleanroom_method_hkappa (raw); \
    } \
    else \
      hk_raw[] = nodata; \
  } \
  boundary ({hk_raw}); \
} while (0)
