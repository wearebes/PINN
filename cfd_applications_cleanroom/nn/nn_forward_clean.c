#include <math.h>

#ifndef NN_WEIGHTS_HEADER
# define NN_WEIGHTS_HEADER "generated/nn_weights_nn27_r128_clean.h"
#endif

#include NN_WEIGHTS_HEADER
#include "nn_forward_clean.h"

static inline double nn_relu_clean (double x) {
  return x > 0.0 ? x : 0.0;
}

void nn_prepare_input_clean (const double raw[RAW_IN], double z[MLP_IN]) {
  for (int i = 0; i < RAW_IN; ++i)
    z[i] = (raw[i] - MEAN[i]) / STD[i];
}

double mlp_hkappa_clean (const double raw[RAW_IN]) {
  double z[MLP_IN];
  double a[MLP_H];
  double b[MLP_H];

  nn_prepare_input_clean (raw, z);

  for (int i = 0; i < MLP_H; ++i) {
    double s = b0[i];
    for (int j = 0; j < MLP_IN; ++j)
      s += W0[i][j] * z[j];
    a[i] = nn_relu_clean (s);
  }

  for (int i = 0; i < MLP_H; ++i) {
    double s = b1[i];
    for (int j = 0; j < MLP_H; ++j)
      s += W1[i][j] * a[j];
    b[i] = nn_relu_clean (s);
  }

  for (int i = 0; i < MLP_H; ++i) {
    double s = b2[i];
    for (int j = 0; j < MLP_H; ++j)
      s += W2[i][j] * b[j];
    a[i] = nn_relu_clean (s);
  }

  for (int i = 0; i < MLP_H; ++i) {
    double s = b3[i];
    for (int j = 0; j < MLP_H; ++j)
      s += W3[i][j] * a[j];
    b[i] = nn_relu_clean (s);
  }

  double out = b4[0];
  for (int j = 0; j < MLP_H; ++j)
    out += W4[0][j] * b[j];
  return out;
}
