#pragma once

#include <math.h>

/* Clean raw27 feature ABI.
 *
 * input_dim=27
 * raw27=[phi9/h, nx9, ny9]
 * target=h*kappa
 * stencil_order=(-1,+1),(0,+1),(+1,+1),(-1,0),(0,0),(+1,0),(-1,-1),(0,-1),(+1,-1)
 * normal_source=central_difference_on_solver_d
 * ghost_requirement=BGHOSTS>=2
 */

static const int NN_CLEAN_OFF_DR[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
static const int NN_CLEAN_OFF_DC[9] = { 1, 1, 1,  0, 0, 0, -1, -1, -1};

static const int NN_CLEAN_D4_MAT[8][4] = {
  { 1,  0,  0,  1},
  { 0, -1,  1,  0},
  {-1,  0,  0, -1},
  { 0,  1, -1,  0},
  { 1,  0,  0, -1},
  {-1,  0,  0,  1},
  { 0,  1,  1,  0},
  { 0, -1, -1,  0},
};

static inline int nn_clean_off_index (int dr, int dc) {
  for (int k = 0; k < 9; ++k)
    if (NN_CLEAN_OFF_DR[k] == dr && NN_CLEAN_OFF_DC[k] == dc)
      return k;
  return -1;
}

static inline void nn_build_raw27_clean (const double D[5][5], double Delta,
                                         double sgn, double raw[27]) {
  for (int k = 0; k < 9; ++k) {
    int a = NN_CLEAN_OFF_DR[k] + 2;
    int b = NN_CLEAN_OFF_DC[k] + 2;
    double dxk = sgn * (D[a + 1][b] - D[a - 1][b]);
    double dyk = sgn * (D[a][b + 1] - D[a][b - 1]);
    double mag = sqrt (dxk*dxk + dyk*dyk);
    if (mag == 0.0)
      mag = 1.0;
    raw[k] = sgn * D[a][b] / Delta;
    raw[9 + k] = dxk / mag;
    raw[18 + k] = dyk / mag;
  }
}

static inline void nn_d4_transform_one_clean (const double raw[27], int m,
                                              double out[27]) {
  int m00 = NN_CLEAN_D4_MAT[m][0], m01 = NN_CLEAN_D4_MAT[m][1];
  int m10 = NN_CLEAN_D4_MAT[m][2], m11 = NN_CLEAN_D4_MAT[m][3];
  for (int dst = 0; dst < 9; ++dst) {
    int dr = NN_CLEAN_OFF_DR[dst], dc = NN_CLEAN_OFF_DC[dst];
    int sdr = m00 * dr + m10 * dc;
    int sdc = m01 * dr + m11 * dc;
    int src = nn_clean_off_index (sdr, sdc);
    double nx = raw[9 + src], ny = raw[18 + src];
    out[dst] = raw[src];
    out[9 + dst] = m00 * nx + m01 * ny;
    out[18 + dst] = m10 * nx + m11 * ny;
  }
}
