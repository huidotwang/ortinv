#ifndef _FDSTENCIL_H
#define _FDSTENCIL_H

#include <stdbool.h>
#include "Grid.h"

typedef struct FDStencil_t FDStencil;
struct FDStencil_t {
  float* coefficients; // size (ndim * nop + 1), value scaled by grid spacing 
  int derivative_order;
  int accuracy_order;
  int nop; // half the size of accuracy order
  int ndim;
  bool opt; // optimized coefficients
};

FDStencil* fdstencil_init(const int derivative_order, const int accuracy_order, const int ndim, const bool opt);

void fdstencil_finalize(FDStencil* fd);

void fdstencil_compute(FDStencil* fd, ...);

#endif
