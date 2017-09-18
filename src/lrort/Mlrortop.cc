/* lowrank P-wave propagation in orthorhombic media */
#include <iostream>
#include <Eigen/Dense>
extern "C" {
#include <complex.h>
#include <fftw3.h>
#include <omp.h>
#include "rsf.h"
#include "Grid.h"
#include "sinc.h"
#include "abcutil.h"
#include "ts_kernel.h"
#include "rsf_wrapper.h"
}
#include "vecmatop.hh"
#include "sample.hh"

#ifdef EIGEN_DEFAULT_DENSE_INDEX_TYPE
#undef EIGEN_DEFAULT_DENSE_INDEX_TYPE
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE std::size_t
#endif

using namespace Eigen;
using std::vector;

static void
wfld3d_inject(float*** uo, float*** ui, int nzo, int nxo, int nyo, int nb)
{
#pragma omp parallel for schedule(dynamic, 1)
  for (int iy = 0; iy < nyo; iy++) {
    for (int ix = 0; ix < nxo; ix++) {
      for (int iz = 0; iz < nzo; iz++) {
        uo[iy + nb][ix + nb][iz + nb] += ui[iy][ix][iz];
      }
    }
  }
  return;
}

int
main(int argc, char* argv[])
{
  bool verb, snap, adj;
  int nz, nx, nt, ns, nr;
  float dz, dx, dt, oz, ox;
  int nz0, nx0, nb;
  float oz0, ox0;
  int nkz, nkx;
  int nzpad, nxpad;

  int ny, ny0, nypad, nky;
  float dy, oy, oy0;

  float ***u1, ***u0;
  float *ws, *wr;

  sf_file file_src = NULL;
  sf_file file_rec = NULL;
  sf_file file_inp = NULL;
  sf_file file_mdl = NULL;
  sf_file file_out = NULL;
  sf_axis az = NULL, ax = NULL, ay = NULL;
  sf_axis az0 = NULL, ax0 = NULL, ay0 = NULL;
  sf_axis at = NULL, as = NULL, ar = NULL;
  pt3d* src3d = NULL;
  pt3d* rec3d = NULL;
  scoef3d cssinc = NULL;
  scoef3d crsinc = NULL;

  float *wi = NULL, *wo = NULL;
  sf_axis ai = NULL, ao = NULL;
  scoef3d cisinc = NULL, cosinc = NULL;
  bool spt = false, rpt = false;
  bool ipt = false, opt = false;

  int seed, npk;
  float eps;
  Eigen::setNbThreads(omp_get_max_threads());

  sf_init(argc, argv);

  if (!sf_getbool("verb", &verb)) verb = false;
  if (!sf_getbool("snap", &snap)) snap = false;
  if (!sf_getbool("adj", &adj)) adj = false;
  if (!sf_getint("nb", &nb)) nb = 4;
  if (!sf_getint("seed", &seed)) seed = time(NULL);
  if (!sf_getfloat("eps", &eps)) eps = 1e-7;
  if (!sf_getint("npk", &npk)) npk = 20;
  if (sf_getstring("sou") != NULL) {
    spt = true;
    if (adj)
      opt = true;
    else
      ipt = true;
  }
  if (sf_getstring("rec") != NULL) {
    rpt = true;
    if (adj)
      ipt = true;
    else
      opt = true;
  }

  file_inp = sf_input("in");
  file_mdl = sf_input("model");
  if (spt) file_src = sf_input("sou");
  if (rpt) file_rec = sf_input("rec");
  file_out = sf_output("out");

  if (ipt)
    at = sf_iaxa(file_inp, 2);
  else
    at = sf_iaxa(file_inp, 4);

  if (spt) as = sf_iaxa(file_src, 2);
  if (rpt) ar = sf_iaxa(file_rec, 2);
  az0 = sf_iaxa(file_mdl, 1);
  ax0 = sf_iaxa(file_mdl, 2);
  ay0 = sf_iaxa(file_mdl, 3);
  nt = sf_n(at);
  dt = sf_d(at); // ot = sf_o(at);
  nz0 = sf_n(az0);
  dz = sf_d(az0);
  oz0 = sf_o(az0);
  nx0 = sf_n(ax0);
  dx = sf_d(ax0);
  ox0 = sf_o(ax0);
  ny0 = sf_n(ay0);
  dy = sf_d(ay0);
  oy0 = sf_o(ay0);

  if (spt) ns = sf_n(as);
  if (rpt) nr = sf_n(ar);
  nz = nz0 + 2 * nb;
  nx = nx0 + 2 * nb;
  ny = ny0 + 2 * nb;
  oz = oz0 - nb * dz;
  ox = ox0 - nb * dx;
  oy = oy0 - nb * dy;
  az = sf_maxa(nz, oz, dz);
  ax = sf_maxa(nx, ox, dx);
  ay = sf_maxa(ny, oy, dy);
  // sf_error("ox=%f ox0=%f oz=%f oz0=%f",ox,ox0,oz,oz0);

  nzpad = kiss_fft_next_fast_size(((nz + 1) >> 1) << 1);
  nkx = nxpad = kiss_fft_next_fast_size(nx);
  nky = nypad = kiss_fft_next_fast_size(ny);
  nkz = nzpad / 2 + 1;
  /* float okx = - 0.5f / dx; */
  float okx = 0.f;
  float oky = 0.f;
  float okz = 0.f;
  float dkx = 1.f / (nxpad * dx);
  float dky = 1.f / (nypad * dy);
  float dkz = 1.f / (nzpad * dz);

  // (1,2,3) = (z,x,y)
  float**** vp = sf_floatalloc4(nz, nx, ny, 3);
  float**** vn = sf_floatalloc4(nz, nx, ny, 3);
  float*** vr = sf_floatalloc3(nz, nx, ny);
  float*** tmparray = sf_floatalloc3(nz0, nx0, ny0);
  sf_floatread(tmparray[0][0], nz0 * nx0 * ny0, file_mdl);
  expand3d(tmparray, vp[0], az0, ax0, ay0, az, ax, ay);
  sf_floatread(tmparray[0][0], nz0 * nx0 * ny0, file_mdl);
  expand3d(tmparray, vp[1], az0, ax0, ay0, az, ax, ay);
  sf_floatread(tmparray[0][0], nz0 * nx0 * ny0, file_mdl);
  expand3d(tmparray, vp[2], az0, ax0, ay0, az, ax, ay);
  sf_floatread(tmparray[0][0], nz0 * nx0 * ny0, file_mdl);
  expand3d(tmparray, vn[0], az0, ax0, ay0, az, ax, ay);
  sf_floatread(tmparray[0][0], nz0 * nx0 * ny0, file_mdl);
  expand3d(tmparray, vn[1], az0, ax0, ay0, az, ax, ay);
  sf_floatread(tmparray[0][0], nz0 * nx0 * ny0, file_mdl);
  expand3d(tmparray, vn[2], az0, ax0, ay0, az, ax, ay);
  sf_floatread(tmparray[0][0], nz0 * nx0 * ny0, file_mdl);
  expand3d(tmparray, vr, az0, ax0, ay0, az, ax, ay);
  free(**tmparray);
  free(*tmparray);
  free(tmparray);

  for (int iy = 0; iy < ny; iy++) {
    for (int ix = 0; ix < nx; ix++) {
      for (int iz = 0; iz < nz; iz++) {
        vp[0][iy][ix][iz] *= vp[0][iy][ix][iz];
        vp[1][iy][ix][iz] *= vp[1][iy][ix][iz];
        vp[2][iy][ix][iz] *= vp[2][iy][ix][iz];
        vn[0][iy][ix][iz] *= vn[0][iy][ix][iz];
        vn[1][iy][ix][iz] *= vn[1][iy][ix][iz];
        vn[2][iy][ix][iz] *= vn[2][iy][ix][iz];
        vr[iy][ix][iz] *= vr[iy][ix][iz] * dt * dt;
      }
    }
  }
  float*** vcomposite = vr;

  float* kk[3];
  kk[0] = sf_floatalloc(nkz);
  kk[1] = sf_floatalloc(nkx);
  kk[2] = sf_floatalloc(nky);

  for (int iky = 0; iky < nky; ++iky) {
    kk[2][iky] = oky + iky * dky;
    if (iky >= nky / 2) kk[2][iky] = (iky - nky) * dky;
    kk[2][iky] *= 2 * SF_PI;
    kk[2][iky] *= kk[2][iky];
  }

  for (int ikx = 0; ikx < nkx; ++ikx) {
    kk[1][ikx] = okx + ikx * dkx;
    if (ikx >= nkx / 2) kk[1][ikx] = (ikx - nkx) * dkx;
    kk[1][ikx] *= 2 * SF_PI;
    kk[1][ikx] *= kk[1][ikx];
  }
  for (int ikz = 0; ikz < nkz; ++ikz) {
    kk[0][ikz] = okz + ikz * dkz;
    kk[0][ikz] *= 2 * SF_PI;
    kk[0][ikz] *= kk[0][ikz];
  }

  if (adj) {
    ai = ar;
    ao = as;
  } else {
    ai = as;
    ao = ar;
  }

  if (opt) {
    sf_oaxa(file_out, ao, 1);
    sf_oaxa(file_out, at, 2);
  } else {
    sf_oaxa(file_out, az0, 1);
    sf_oaxa(file_out, ax0, 2);
    sf_oaxa(file_out, ay0, 3);
    sf_oaxa(file_out, at, 4);
  }
  sf_fileflush(file_out, NULL);

  if (spt) {
    src3d = pt3dalloc1(ns);
    pt3dread1(file_src, src3d, ns, 3);
    cssinc = sinc3d_make(ns, src3d, az, ax, ay);
    ws = sf_floatalloc(ns);
    if (adj) {
      cosinc = cssinc;
      wo = ws;
    } else {
      cisinc = cssinc;
      wi = ws;
    }
  }
  if (rpt) {
    rec3d = pt3dalloc1(nr);
    pt3dread1(file_rec, rec3d, nr, 3);
    crsinc = sinc3d_make(nr, rec3d, az, ax, ay);
    wr = sf_floatalloc(nr);
    if (adj) {
      cisinc = crsinc;
      wi = wr;
    } else {
      cosinc = crsinc;
      wo = wr;
    }
  }

  // lowrank decomposition
  size_t m = nz * nx * ny;
  size_t n = nkz * nkx * nky;
  void* ort_lrparam = lrdecomp_new(m, n);
  lrdecomp_init(ort_lrparam, seed, npk, eps, dt, nkz, nkx, nky, kk[0], kk[1], kk[2],
                vp[0][0][0], vp[1][0][0], vp[2][0][0], vn[0][0][0], vn[1][0][0],
                vn[2][0][0]);
  // lrmat* ort_lrmat = lrdecomp_compute(ort_lrparam, m, n, seed, npk, eps);
  lrmat* ort_lrmat = lrdecomp_compute(ort_lrparam);
  int nrank = ort_lrmat->nrank;
  float** lft = sf_floatalloc2(m, nrank);
  float** rht = sf_floatalloc2(n, nrank);
  memcpy(lft[0], ort_lrmat->lft_data, sizeof(float) * m * nrank);
  memcpy(rht[0], ort_lrmat->rht_data, sizeof(float) * n * nrank);
  lrdecomp_delete(ort_lrparam, ort_lrmat);

  u0 = sf_floatalloc3(nz, nx, ny);
  u1 = sf_floatalloc3(nz, nx, ny);

  /* float* rwave = (float*)fftwf_malloc(nzpad * nxpad * nypad * sizeof(float));
   */
  float* rwavem = (float*)fftwf_malloc(nzpad * nxpad * nypad * sizeof(float));
  fftwf_complex* cwave =
    (fftwf_complex*)fftwf_malloc(nkz * nkx * nky * sizeof(fftwf_complex));
  fftwf_complex* cwavem =
    (fftwf_complex*)fftwf_malloc(nkz * nkx * nky * sizeof(fftwf_complex));

  /* boundary conditions */
  float*** ucut = NULL;
  if (!(ipt && opt)) ucut = sf_floatalloc3(nz0, nx0, ny0);
  float* damp = damp_make(nb);

  float wt = 1.0 / (nxpad * nzpad * nypad);
  fftwf_plan forward_plan;
  fftwf_plan inverse_plan;
  fftwf_init_threads();
  fftwf_plan_with_nthreads(omp_get_max_threads());
  forward_plan =
    fftwf_plan_dft_r2c_3d(nypad, nxpad, nzpad, rwavem, cwave, FFTW_MEASURE);
  fftwf_plan_with_nthreads(omp_get_max_threads());
  inverse_plan =
    fftwf_plan_dft_c2r_3d(nypad, nxpad, nzpad, cwavem, rwavem, FFTW_MEASURE);

  int itb, ite, itc;
  if (adj) {
    itb = nt - 1;
    ite = -1;
    itc = -1;
  } else {
    itb = 0;
    ite = nt;
    itc = 1;
  }

  if (adj) {
    for (int it = 0; it < nt; it++) {
      if (opt)
        sf_floatwrite(wo, sf_n(ao), file_out);
      else
        sf_floatwrite(ucut[0][0], nz0 * nx0 * ny0, file_out);
    }
    sf_seek(file_out, 0, SEEK_SET);
  }

  float*** ptrtmp = NULL;
  memset(u0[0][0], 0, sizeof(float) * nz * nx * ny);
  memset(u1[0][0], 0, sizeof(float) * nz * nx * ny);

  for (int it = itb; it != ite; it += itc) {
    if (verb) sf_warning("it = %d;", it);
    double tic = omp_get_wtime();

    if (ipt) {
      if (adj)
        sf_seek(file_inp, (off_t)(it) * sizeof(float) * sf_n(ai), SEEK_SET);
      sf_floatread(wi, sf_n(ai), file_inp);
    } else {
      if (adj)
        sf_seek(file_inp, (off_t)(it) * sizeof(float) * nz0 * nx0 * ny0,
                SEEK_SET);
      sf_floatread(ucut[0][0], nz0 * nx0 * ny0, file_inp);
    }

    lr_fft_stepforward(u0, u1, rwavem, cwave, cwavem, lft, rht, forward_plan,
                       inverse_plan, nz, nx, ny, nzpad, nxpad, nypad, nkz, nkx,
                       nky, nrank, wt, adj);
    if (ipt) /* sinc3d_inject_with_vv(u0, wi, cisinc, vcomposite); */
      sinc3d_inject(u0, wi, cisinc);
    else
      wfld3d_inject(u0, ucut, nz0, nx0, ny0, nb);
#if 0
    /* apply absorbing boundary condition: E \times u@n+1 */
    damp3d_apply(u1, damp, nz, nx, ny, nb);
    damp3d_apply(u0, damp, nz, nx, ny, nb);
#endif
    /* loop over pointers */
    ptrtmp = u0;
    u0 = u1;
    u1 = ptrtmp;

    if (opt) {
      if (adj)
        sf_seek(file_out, (off_t)(it) * sizeof(float) * sf_n(ao), SEEK_SET);
      sinc3d_extract(u0, wo, cosinc);
      sf_floatwrite(wo, sf_n(ao), file_out);
    } else {
      if (adj)
        sf_seek(file_out, (off_t)(it) * sizeof(float) * nz0 * nx0 * ny0,
                SEEK_SET);
      wwin3d(ucut, u0, nz0, nx0, ny0, nb);
      sf_floatwrite(ucut[0][0], nz0 * nx0 * ny0, file_out);
    }

    double toc = omp_get_wtime();
    if (verb) fprintf(stderr, " clock = %lf;", toc - tic);
  } /* END OF FORWARD TIME LOOP */

  return 0;
}
