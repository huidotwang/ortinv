#ifndef _RSF_WRAPPER_H
#define _RSF_WRAPPER_H

float** rsf_flt2d_wrapper(float* buf, size_t n1, size_t n2);
float*** rsf_flt3d_wrapper(float* buf, size_t n1, size_t n2, size_t n3);
float**** rsf_flt4d_wrapper(float* buf, size_t n1, size_t n2, size_t n3,
                            size_t n4);
#endif
