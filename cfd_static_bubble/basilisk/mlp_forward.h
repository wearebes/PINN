#pragma once

#include <math.h>

#include "mlp_weights.h"


static inline float relu(float x) {
  return x > 0.0f ? x : 0.0f;
}


static inline double mlp_hkappa(const double raw[MLP_IN]) {
  float z[MLP_IN];
  float a[MLP_H];
  float b[MLP_H];

  for (int i = 0; i < MLP_IN; ++i)
    z[i] = (float)((raw[i] - MEAN[i]) / STD[i]);

  for (int i = 0; i < MLP_H; ++i) {
    float s = (float)b0[i];
    for (int j = 0; j < MLP_IN; ++j)
      s += (float)W0[i][j] * z[j];
    a[i] = relu(s);
  }

  for (int i = 0; i < MLP_H; ++i) {
    float s = (float)b1[i];
    for (int j = 0; j < MLP_H; ++j)
      s += (float)W1[i][j] * a[j];
    b[i] = relu(s);
  }

  for (int i = 0; i < MLP_H; ++i) {
    float s = (float)b2[i];
    for (int j = 0; j < MLP_H; ++j)
      s += (float)W2[i][j] * b[j];
    a[i] = relu(s);
  }

  for (int i = 0; i < MLP_H; ++i) {
    float s = (float)b3[i];
    for (int j = 0; j < MLP_H; ++j)
      s += (float)W3[i][j] * a[j];
    b[i] = relu(s);
  }

  float out = (float)b4[0];
  for (int j = 0; j < MLP_H; ++j)
    out += (float)W4[0][j] * b[j];
  return (double)out;
}
