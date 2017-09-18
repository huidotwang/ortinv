#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>


#include "FDStencil.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

FDStencil*
fdstencil_init(const int _derivative_order, const int _accuracy_order, const int _ndim, const bool _opt)
{
  FDStencil* fd = malloc(sizeof(*fd));
  fd->derivative_order = _derivative_order;
  fd->accuracy_order = _accuracy_order;
  fd->nop = _accuracy_order / 2;
  fd->ndim = _ndim;
  fd->opt = _opt;
  fd->coefficients = malloc(sizeof(float) * (fd->ndim * fd->nop + 1));
  return fd;
}

void
fdstencil_finalize(FDStencil* fd)
{
  assert(fd);
  assert(fd->coefficients);
  free(fd->coefficients);
  free(fd);
}

static size_t
factorial(int n)
{
  size_t result = 1;
  for (int i = 1; i <= n; ++i) result *= i;
  return result;
}

static float*
normal_fdcoef(int nop, int d_order)
{
  float* cc = calloc(nop + 1, sizeof(float));
  float* bb = calloc(nop + 1, sizeof(float));
  size_t halfN_fact = factorial(nop);
  halfN_fact *= halfN_fact;
  for (int n = 1; n <= nop; n++) {
    cc[n] = -2.f / (n * n) * cos(n * M_PI) * halfN_fact / factorial(nop + n) /
            factorial(nop - n);
    bb[n] = cc[n] * n / 2.f;
  }
  if (d_order == 1)
    return bb;
  else
    return cc;
}

static float*
optimal_fdcoef(int nop, int d_order)
{
  if (d_order == 2) {
    float* opt_c[11];
    float opt_c1[2] = { 0.f, 1.f };                                 // nop=1
    float opt_c2[3] = { 0.f, 1.369074f, -0.09266816f };             // nop=2
    float opt_c3[4] = { 0.f, 1.573661f, -0.1820268f, 0.01728053f }; // nop=3
    float opt_c4[5] = { 0.f, 1.700010f, -0.2554615f, 0.04445392f,
                        -0.004946851f }; // nop=4
    float opt_c5[6] = { 0.f,         1.782836f,    -0.3124513f,
                        0.07379487f, -0.01532122f, 0.001954439f }; // nop=5
    float opt_c6[7] = { 0.f,          1.837023f,    -0.3538895f,   0.09978343f,
                        -0.02815486f, 0.006556587f, -0.0009405699f }; // nop=6
    float opt_c7[8] = {
      0.f,          1.874503f,   -0.3845794f,   0.1215162f,
      -0.04121749f, 0.01295522f, -0.003313813f, 0.0005310053f
    }; // nop=7
    float opt_c8[9] = { 0.f,           1.901160f,    -0.4074304f,
                        0.1390909f,    -0.05318775f, 0.02004823f,
                        -0.006828249f, 0.001895771f, -0.0003369052f }; // nop=8
    float opt_c9[10] = { 0.f,          1.919909f,    -0.4240446f,
                         0.1526043f,   -0.06322328f, 0.02676005f,
                         -0.01080739f, 0.003907747f, -0.001158024f,
                         0.0002240247f }; // nop=9
    float opt_c10[11] = { 0.f,           1.918204f,      -0.4225858f,
                          0.1514992f,    -0.06249474f,   0.02637196f,
                          -0.01066631f,  0.003915625f,   -0.001219872f,
                          0.0002863976f, -0.00003744830f }; // nop=10
    opt_c[1] = opt_c1;
    opt_c[2] = opt_c2;
    opt_c[3] = opt_c3;
    opt_c[4] = opt_c4;
    opt_c[5] = opt_c5;
    opt_c[6] = opt_c6;
    opt_c[7] = opt_c7;
    opt_c[8] = opt_c8;
    opt_c[9] = opt_c9;
    opt_c[10] = opt_c10;
    float* cc = malloc((nop + 1) * sizeof(float));
    memcpy(cc, opt_c[nop], sizeof(float) * (nop + 1));
    return cc;
  } else {
    float* opt_b[7];
    float opt_b1[2] = { 0.0f, 0.5f };                                // nop=1
    float opt_b2[3] = { 0.0f, 0.67880327, -0.08962729 };             // nop=2
    float opt_b3[4] = { 0.0f, 0.77793115, -0.17388691, 0.02338713 }; // nop=3
    float opt_b4[5] = { 0.0f, 0.84149635, -0.24532989, 0.06081891,
                        -0.00839807 }; // nop=4
    float opt_b5[6] = { 0.0f,       0.88414717,  -0.30233648,
                        0.10275057, -0.02681517, 0.00398089 }; // nop=5
    float opt_b6[7] = { 0.0f,        0.91067892, -0.34187892, 0.13833962,
                        -0.04880710, 0.01302148, -0.00199047 }; // nop=6

    opt_b[1] = opt_b1;
    opt_b[2] = opt_b2;
    opt_b[3] = opt_b3;
    opt_b[4] = opt_b4;
    opt_b[5] = opt_b5;
    opt_b[6] = opt_b6;
    float* bb = malloc((nop + 1) * sizeof(float));
    memcpy(bb, opt_b[nop], sizeof(float) * (nop + 1));
    return bb;
  }
}

void
fdstencil_compute(FDStencil* fd, ...)
{
  int nop = fd->nop;
  int d_order = fd->derivative_order;
  int ndim = fd->ndim;
  float d2[ndim];
  float d1[ndim];
  float* ccc;
  float* bbb;
  va_list ap;
  va_start(ap, fd);
  for (int i = 0; i < ndim; ++i) {
    d1[i] = sf_d(va_arg(ap, sf_axis));
    d2[i] = d1[i] * d1[i];
  }
  va_end(ap);

  if (fd->derivative_order == 2) {
    if (fd->opt)
      ccc = optimal_fdcoef(nop, 2);
    else
      ccc = normal_fdcoef(nop, d_order);
    fd->coefficients[0] = 0.f;
    for (int idim = 0; idim < ndim; ++idim) {
      for (int ii = 1; ii <= nop; ii++) {
        fd->coefficients[idim * nop + ii] = ccc[ii] / d2[idim];
        fd->coefficients[0] += fd->coefficients[idim * nop + ii];
      }
    }
    fd->coefficients[0] *= -2.0f;
  } else {
    if (fd->opt && nop <= 6)
      bbb = optimal_fdcoef(nop, 1);
    else
      bbb = normal_fdcoef(nop, d_order);
    fd->coefficients[0] = 0.f;
    for (int idim = 0; idim < ndim; ++idim)
      for (int ii = 1; ii <= nop; ii++)
        fd->coefficients[idim * nop + ii] = bbb[ii] / d1[idim];
  }
}
