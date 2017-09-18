#include "ts_kernel.h"

static inline bool
is_zero(float a)
{
  return fpclassify(a) == FP_ZERO;
}

void
lr_fft_stepforward(float*** u0, float*** u1, float* rwavem,
                   fftwf_complex* cwave, fftwf_complex* cwavem, float** lft,
                   float** rht, fftwf_plan forward_plan,
                   fftwf_plan inverse_plan, int nz, int nx, int ny, int nzpad,
                   int nxpad, int nypad, int nkz, int nkx, int nky, int nrank,
                   float wt, bool adj)
{
#pragma omp parallel for schedule(dynamic, 1)
  for (int iy = 0; iy < nypad; iy++) {
    memset(&rwavem[iy * nzpad * nxpad], 0, sizeof(float) * nzpad * nxpad);
    memset(&cwave[iy * nkz * nkx], 0, sizeof(fftwf_complex) * nkz * nkx);
    memset(&cwavem[iy * nkz * nkx], 0, sizeof(fftwf_complex) * nkz * nkx);
  }

  if (adj) { /* adjoint modeling */
    for (int im = 0; im < nrank; im++) {
#pragma omp parallel for schedule(dynamic, 1)
      /* rwavem = L^T_i \schur_dot rwave */
      for (int k = 0; k < ny; k++) {
        for (int j = 0; j < nx; j++) {
          for (int i = 0; i < nz; i++) {
            int ii = (k * nx + j) * nz + i;
            int jj = (k * nxpad + j) * nzpad + i;
            rwavem[jj] = lft[im][ii] * u1[k][j][i];
          }
        }
      }
      /* --- 3D forward Fourier transform ---*/
      fftwf_execute(forward_plan);

#pragma omp parallel for schedule(dynamic, 1)
      /* cwavem += R^T_i \schur_dot cwave */
      for (int k = 0; k < nky; k++) {
        int stride = nkz * nkx;
        for (int ii = 0; ii < stride; ii++) {
          int idx = k * stride + ii;
          cwavem[idx] += rht[im][idx] * cwave[idx];
        }
      }
    }
    /* --- 3D backward Fourier transform ---*/
    fftwf_execute(inverse_plan);

#pragma omp parallel for schedule(dynamic, 1)
    for (int k = 0; k < ny; k++) {
      for (int j = 0; j < nx; j++) {
        for (int i = 0; i < nz; i++) {
          int jj = (k * nxpad + j) * nzpad + i;
          u0[k][j][i] = 2.0f * u1[k][j][i] - u0[k][j][i];
          /* FFT normalization */
          u0[k][j][i] += rwavem[jj] * wt;
        }
      }
    }

  } else { /* forward modeling */
#pragma omp parallel for schedule(dynamic, 1)
    for (int k = 0; k < ny; k++) {
      for (int j = 0; j < nx; j++) {
        for (int i = 0; i < nz; i++) {
          int jj = (k * nxpad + j) * nzpad + i;
          u0[k][j][i] = 2.0f * u1[k][j][i] - u0[k][j][i];
          rwavem[jj] = u1[k][j][i];
        }
      }
    }

    /* --- 3D forward Fourier transform ---*/
    fftwf_execute(forward_plan);

    for (int im = 0; im < nrank; im++) {
/* element-wise vector multiplication: u@t(kz,kx) * M3(im,:) */
#pragma omp parallel for schedule(dynamic, 1)
      for (int k = 0; k < nky; k++) {
        int stride = nkz * nkx;
        for (int ii = 0; ii < stride; ii++) {
          int idx = k * stride + ii;
          cwavem[idx] = rht[im][idx] * cwave[idx];
        }
      }

      /* --- 3D backward Fourier transform ---*/
      fftwf_execute(inverse_plan);

/* element-wise vector multiplication: M1(:,im) * u@t(z,x) */
#pragma omp parallel for schedule(dynamic, 1)
      for (int k = 0; k < ny; k++) {
        for (int j = 0; j < nx; j++) {
          for (int i = 0; i < nz; i++) {
            int ii = (k * nx + j) * nz + i;
            int jj = (k * nxpad + j) * nzpad + i;
            /* FFT normalization \times wt */
            u0[k][j][i] += lft[im][ii] * rwavem[jj] * wt;
          }
        }
      }
    }
  }
  return;
}

// Gradient calculation throught generalized pseudo-spectral propagator
void
compute_gradients_vpn(float*** us, float*** ur, float*** gradients[],
                      float* rwavem, fftwf_complex* cwave,
                      fftwf_complex* cwavem, fftwf_plan forward_plan,
                      fftwf_plan inverse_plan, float*** vp[], float*** vpn[],
                      float*** vpr, float* kk[], int nz, int nx, int ny,
                      int nzpad, int nxpad, int nypad, int nkz, int nkx,
                      int nky, float wt)
{
#pragma omp parallel for schedule(dynamic, 1)
  for (int iy = 0; iy < nypad; iy++) {
    memset(&rwavem[iy * nzpad * nxpad], 0, sizeof(float) * nzpad * nxpad);
    memset(&cwave[iy * nkz * nkx], 0, sizeof(fftwf_complex) * nkz * nkx);
    memset(&cwavem[iy * nkz * nkx], 0, sizeof(fftwf_complex) * nkz * nkx);
  }
#pragma omp parallel for schedule(dynamic, 1)
  for (int k = 0; k < ny; k++) {
    for (int j = 0; j < nx; j++) {
      for (int i = 0; i < nz; i++) {
        int jj = (k * nxpad + j) * nzpad + i;
        rwavem[jj] = us[k][j][i];
      }
    }
  }
  fftwf_execute(forward_plan);

/* dF/dVp1^2 : k1^2 */
#pragma omp parallel for schedule(dynamic, 1)
  for (int iky = 0; iky < nky; iky++) {
    for (int ikx = 0; ikx < nkx; ikx++) {
      for (int ikz = 0; ikz < nkz; ikz++) {
        int idx = (iky * nkx + ikx) * nkz + ikz;
        cwavem[idx] = cwave[idx] * kk[0][ikz];
      }
    }
  }
  fftwf_execute(inverse_plan);
#pragma omp parallel for schedule(dynamic, 1)
  for (int k = 0; k < ny; k++) {
    for (int j = 0; j < nx; j++) {
      for (int i = 0; i < nz; i++) {
        int jj = (k * nxpad + j) * nzpad + i;
        gradients[0][k][j][i] += wt * rwavem[jj] * ur[k][j][i];
      }
    }
  }
/* dF/dVp2^2 : k2^2 */
#pragma omp parallel for schedule(dynamic, 1)
  for (int iky = 0; iky < nky; iky++) {
    for (int ikx = 0; ikx < nkx; ikx++) {
      for (int ikz = 0; ikz < nkz; ikz++) {
        int idx = (iky * nkx + ikx) * nkz + ikz;
        cwavem[idx] = cwave[idx] * kk[1][ikx];
      }
    }
  }
  fftwf_execute(inverse_plan);
#pragma omp parallel for schedule(dynamic, 1)
  for (int k = 0; k < ny; k++) {
    for (int j = 0; j < nx; j++) {
      for (int i = 0; i < nz; i++) {
        int jj = (k * nxpad + j) * nzpad + i;
        gradients[1][k][j][i] += wt * rwavem[jj] * ur[k][j][i];
      }
    }
  }

/* dF/dVp3^2 : k3^2 */
#pragma omp parallel for schedule(dynamic, 1)
  for (int iky = 0; iky < nky; iky++) {
    for (int ikx = 0; ikx < nkx; ikx++) {
      for (int ikz = 0; ikz < nkz; ikz++) {
        int idx = (iky * nkx + ikx) * nkz + ikz;
        cwavem[idx] = cwave[idx] * kk[2][iky];
      }
    }
  }
  fftwf_execute(inverse_plan);
#pragma omp parallel for schedule(dynamic, 1)
  for (int k = 0; k < ny; k++) {
    for (int j = 0; j < nx; j++) {
      for (int i = 0; i < nz; i++) {
        int jj = (k * nxpad + j) * nzpad + i;
        gradients[2][k][j][i] += wt * rwavem[jj] * ur[k][j][i];
      }
    }
  }

/* dF/dVpx^2 : (k1k2/k)^2 */
/* dF/dVpz^2 : (k1k2/k)^2 */
/* dF/dVa2^2 : (k1k2/k)^2 */
#pragma omp parallel for schedule(dynamic, 1)
  for (int iky = 0; iky < nky; iky++) {
    for (int ikx = 0; ikx < nkx; ikx++) {
      for (int ikz = 0; ikz < nkz; ikz++) {
        int idx = (iky * nkx + ikx) * nkz + ikz;
        float k2sum = kk[2][iky] + kk[1][ikx] + kk[0][ikz];
        float ratio = 0.f;
        if (!is_zero(k2sum)) ratio = (kk[1][ikx] * kk[0][ikz]) / k2sum;
        cwavem[idx] = cwave[idx] * ratio;
      }
    }
  }
  fftwf_execute(inverse_plan);
#pragma omp parallel for schedule(dynamic, 1)
  for (int k = 0; k < ny; k++) {
    for (int j = 0; j < nx; j++) {
      for (int i = 0; i < nz; i++) {
        int jj = (k * nxpad + j) * nzpad + i;
        float value = wt * rwavem[jj] * vpr[k][j][i] * ur[k][j][i];
        gradients[0][k][j][i] -= value * (vp[1][k][j][i] - vpn[1][k][j][i]);
        gradients[1][k][j][i] -= value * vp[0][k][j][i];
        /* gradients[5][k][j][i] += value * vpa[2][k][j][i] * 2.0; */
        gradients[4][k][j][i] += value * vp[0][k][j][i];
      }
    }
  }

/* dF/dVpy^2 : (k1k3/k)^2 */
/* dF/dVpz^2 : (k1k3/k)^2 */
/* dF/dVp1^2 : (k1k3/k)^2 */
#pragma omp parallel for schedule(dynamic, 1)
  for (int iky = 0; iky < nky; iky++) {
    for (int ikx = 0; ikx < nkx; ikx++) {
      for (int ikz = 0; ikz < nkz; ikz++) {
        int idx = (iky * nkx + ikx) * nkz + ikz;
        float k2sum = kk[2][iky] + kk[1][ikx] + kk[0][ikz];
        float ratio = 0.f;
        /* if (!is_zero(k2sum)) ratio = (kk[1][ikx] * kk[0][ikz]) / k2sum; */
        if (!is_zero(k2sum)) ratio = (kk[2][iky] * kk[0][ikz]) / k2sum;
        cwavem[idx] = cwave[idx] * ratio;
      }
    }
  }
  fftwf_execute(inverse_plan);
#pragma omp parallel for schedule(dynamic, 1)
  for (int k = 0; k < ny; k++) {
    for (int j = 0; j < nx; j++) {
      for (int i = 0; i < nz; i++) {
        int jj = (k * nxpad + j) * nzpad + i;
        float value = wt * rwavem[jj] * vpr[k][j][i] * ur[k][j][i];
        gradients[0][k][j][i] -= value * (vp[2][k][j][i] - vpn[0][k][j][i]);
        gradients[2][k][j][i] -= value * vp[0][k][j][i];
        /* gradients[4][k][j][i] += value * vpa[1][k][j][i] * 2.0f; */
        gradients[3][k][j][i] += value * vp[0][k][j][i];
      }
    }
  }

/* dF/dVpx^2 : (k2k3/k)^2 */
/* dF/dVpy^2 : (k2k3/k)^2 */
/* dF/dVp3^2 : (k2k3/k)^2 */
#pragma omp parallel for schedule(dynamic, 1)
  for (int iky = 0; iky < nky; iky++) {
    for (int ikx = 0; ikx < nkx; ikx++) {
      for (int ikz = 0; ikz < nkz; ikz++) {
        int idx = (iky * nkx + ikx) * nkz + ikz;
        float k2sum = kk[2][iky] + kk[1][ikx] + kk[0][ikz];
        float ratio = 0.f;
        if (!is_zero(k2sum)) ratio = (kk[1][ikx] * kk[2][iky]) / k2sum;
        cwavem[idx] = cwave[idx] * ratio;
      }
    }
  }
  fftwf_execute(inverse_plan);
#pragma omp parallel for schedule(dynamic, 1)
  for (int k = 0; k < ny; k++) {
    for (int j = 0; j < nx; j++) {
      for (int i = 0; i < nz; i++) {
        int jj = (k * nxpad + j) * nzpad + i;
        float value = wt * rwavem[jj] * vpr[k][j][i] * ur[k][j][i];
        gradients[1][k][j][i] -= value * (vp[2][k][j][i] - vpn[2][k][j][i]);
        gradients[2][k][j][i] -= value * vp[1][k][j][i];
        /* gradients[3][k][j][i] += value * vpa[0][k][j][i] * 2.0f; */
        gradients[5][k][j][i] += value * vp[1][k][j][i];
      }
    }
  }
  return;
}

#if 1
void
compute_gradients_vpn_fd(float*** us, float*** ur, float*** gradients[],
                         float*** vp[], float*** vn[], float*** vr, int nzpad,
                         int nxpad, int nypad, float* czz, float* cxx,
                         float* cyy, int nop)
{
#if 0
  float c0 = fdcoef_d2[0];
  float* cz = &fdcoef_d2[0];
  float* cx = &fdcoef_d2[nop];
  float* cy = &fdcoef_d2[nop + nop];
#endif
  float*** vpz = vp[0];
  float*** vpx = vp[1];
  float*** vpy = vp[2];
  float*** vn1 = vn[0];
  float*** vn2 = vn[1];
  float*** vn3 = vn[2];
  float*** g_vpz = gradients[0];
  float*** g_vpx = gradients[1];
  float*** g_vpy = gradients[2];
  float*** g_vn1 = gradients[3];
  float*** g_vn2 = gradients[4];
  float*** g_vn3 = gradients[5];
#pragma omp parallel for schedule(static, 1)
  for (int iy = nop; iy < nypad - nop; iy++) {
    for (int ix = nop; ix < nxpad - nop; ix++) {
      for (int iz = nop; iz < nzpad - nop; iz++) {
        float us_dzz = us[iy][ix][iz] * czz[0];
        float us_dxx = us[iy][ix][iz] * cxx[0];
        float us_dyy = us[iy][ix][iz] * cyy[0];
        for (int iop = 1; iop <= nop; iop++) {
          us_dzz += (us[iy][ix][iz - iop] + us[iy][ix][iz + iop]) * czz[iop];
          us_dxx += (us[iy][ix - iop][iz] + us[iy][ix + iop][iz]) * cxx[iop];
          us_dyy += (us[iy - iop][ix][iz] + us[iy + iop][ix][iz]) * cyy[iop];
        }
        float dUs_vpz = 0.f;
        float dUs_vpx = 0.f;
        float dUs_vpy = 0.f;
        dUs_vpz -= us_dzz;
        dUs_vpz += us_dxx * (vpx[iy][ix][iz] - vn2[iy][ix][iz]) * vr[iy][ix][iz];
        dUs_vpz += us_dyy * (vpy[iy][ix][iz] - vn1[iy][ix][iz]) * vr[iy][ix][iz];
        dUs_vpx -= us_dxx;
        dUs_vpx += us_dyy * (vpy[iy][ix][iz] - vn3[iy][ix][iz]) * vr[iy][ix][iz];
        dUs_vpx += us_dzz * vpz[iy][ix][iz] * vr[iy][ix][iz];
        dUs_vpy -= us_dyy;
        dUs_vpy += us_dxx * vpx[iy][ix][iz] * vr[iy][ix][iz];
        dUs_vpy += us_dzz * vpz[iy][ix][iz] * vr[iy][ix][iz];
        g_vpz[iy][ix][iz] += dUs_vpz * ur[iy][ix][iz];
        g_vpx[iy][ix][iz] += dUs_vpx * ur[iy][ix][iz];
        g_vpy[iy][ix][iz] += dUs_vpy * ur[iy][ix][iz];
#if 0
        float lap = u1[iy][ix][iz] * c0;
        for (int iop = 1; iop <= nop; iop++) {
          lap += (us[iy][ix][iz - iop] + us[iy][ix][iz + iop]) * cz[iop] +
                 (us[iy][ix - iop][iz] + us[iy][ix + iop][iz]) * cx[iop] +
                 (us[iy - iop][ix][iz] + us[iy + iop][ix][iz]) * cy[iop];
        }
        u0[iy][ix][iz] =
          2. * u1[iy][ix][iz] - u0[iy][ix][iz] + vel[iy][ix][iz] * lap;
#endif
      }
    }
  }
  return;
}
#endif
