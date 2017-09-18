#ifndef _TS_KERNEL_H
#define _TS_KERNEL_H
#include <string.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <stdbool.h>
#define compute_gradients compute_gradients_vpn

void compute_gradients_vpn(float*** us, float*** ur, float*** gradients[],
                           float* rwavem, fftwf_complex* cwave,
                           fftwf_complex* cwavem, fftwf_plan forward_plan,
                           fftwf_plan inverse_plan, float*** vp[],
                           float*** vn[], float*** vpr, float* kk[], int nz,
                           int nx, int ny, int nzpad, int nxpad, int nypad,
                           int nkz, int nkx, int nky, float wt);

void lr_fft_stepforward(float*** u0, float*** u1, float* rwavem,
                        fftwf_complex* cwave, fftwf_complex* cwavem,
                        float** lft, float** rht, fftwf_plan forward_plan,
                        fftwf_plan inverse_plan, int nz, int nx, int ny,
                        int nzpad, int nxpad, int nypad, int nkz, int nkx,
                        int nky, int nrank, float wt, bool adj);
void
compute_gradients_vpn_fd(float*** us, float*** ur, float*** gradients[],
                         float*** vp[], float*** vn[], float*** vr, int nzpad,
                         int nxpad, int nypad, float* czz, float* cxx,
                         float* cyy, int nop);
#endif
