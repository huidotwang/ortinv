#include <stdlib.h>
#include "rsf_wrapper.h"

float**
rsf_flt2d_wrapper(float* buf, size_t n1, size_t n2)
{
  float** ptr = (float**)malloc(n2 * sizeof(float*));
  ptr[0] = buf;
  for (size_t i2 = 1; i2 < n2; i2++) ptr[i2] = ptr[0] + i2 * n1;
  return ptr;
}

float***
rsf_flt3d_wrapper(float* buf, size_t n1, size_t n2, size_t n3)
{
  float*** ptr = (float***)malloc(n3 * sizeof(float**));
  ptr[0] = rsf_flt2d_wrapper(buf, n1, n2 * n3);
  for (size_t i3 = 1; i3 < n3; i3++) ptr[i3] = ptr[0] + i3 * n2;
  return ptr;
}

float****
rsf_flt4d_wrapper(float* buf, size_t n1, size_t n2, size_t n3, size_t n4)
{
  float**** ptr = (float****)malloc(n4 * sizeof(float***));
  ptr[0] = rsf_flt3d_wrapper(buf, n1, n2, n3 * n4);
  for (size_t i4 = 1; i4 < n4; i4++) ptr[i4] = ptr[0] + i4 * n3;
  return ptr;
}
