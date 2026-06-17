#include <math.h>
#include <stdio.h>

#include "mlp_forward.h"
#include "parity_fixtures.h"


int main(void) {
  double max_rel_error = 0.0;
  for (int i = 0; i < PARITY_FIXTURE_COUNT; ++i) {
    double got = mlp_hkappa(PARITY_RAW27[i]);
    double want = PARITY_EXPECTED_HKAPPA[i];
    double denom = fabs(want) > 1e-12 ? fabs(want) : 1.0;
    double rel = fabs(got - want) / denom;
    if (rel > max_rel_error)
      max_rel_error = rel;
  }

  printf("{\"max_rel_error\": %.12g}\n", max_rel_error);
  return max_rel_error < 1e-6 ? 0 : 1;
}
