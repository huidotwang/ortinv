/* separable qP wave propagation in orthorhombic media */
// l2-norm of data difference
#include <iostream>
#include <Eigen/Dense>
#include <mpi.h>
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

int
main(int argc, char* argv[])
{
  bool verb, snap;
  int nz, nx, nt, ns, nr;
  float dz, dx, dt, oz, ox;
  int nz0, nx0, nb;
  float oz0, ox0;
  int nkz, nkx;
  int nzpad, nxpad;
  
  int ny, ny0, nypad, nky;
  float dy, oy, oy0;
  
  float ***u1, ***u0;
  float **ws, **wr;
  
  sf_file file_src = NULL;
  sf_file file_rec = NULL;
  sf_file file_inp = NULL;
  sf_file file_wfl = NULL;
  sf_file file_mdl = NULL;
  sf_file file_dat = NULL;
  sf_axis az = NULL, ax = NULL, ay = NULL;
  sf_axis az0 = NULL, ax0 = NULL, ay0 = NULL;
  sf_axis at = NULL, as = NULL, ar = NULL;
  pt3d** src3d = NULL;
  pt3d** rec3d = NULL;
  scoef3d* cssinc = NULL;
  scoef3d* crsinc = NULL;
  
  int seed, npk;
  float eps;
  
  Eigen::setNbThreads(omp_get_max_threads());
  int nprocs, myrank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  double stime = MPI_Wtime();
  
  sf_init(argc, argv);
  
  if (!sf_getbool("verb", &verb)) verb = false;
  if (!sf_getbool("snap", &snap)) snap = false;
  if (!sf_getint("nb", &nb)) nb = 4;
  if (!sf_getint("seed", &seed)) seed = time(NULL);
  if (!sf_getfloat("eps", &eps)) eps = 1e-7;
  if (!sf_getint("npk", &npk)) npk = 20;
  
  file_inp = sf_input("input");
  file_mdl = sf_input("model");
  file_src = sf_input("sou");
  file_rec = sf_input("rec");
  file_dat = sf_output("output");
  if (sf_getstring("owfl") != NULL) {
    snap = true;
    file_wfl = sf_output("owfl");
  }
  
  at = sf_iaxa(file_inp, 2);
  as = sf_iaxa(file_src, 2);
  ar = sf_iaxa(file_rec, 2);
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
  
  ns = sf_n(as);
  nr = sf_n(ar);
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
  float*** vpz = sf_floatalloc3(nz, nx, ny);
  float*** vpx = sf_floatalloc3(nz, nx, ny);
  float*** vpy = sf_floatalloc3(nz, nx, ny);
  float*** vn1 = sf_floatalloc3(nz, nx, ny);
  float*** vn2 = sf_floatalloc3(nz, nx, ny);
  float*** vn3 = sf_floatalloc3(nz, nx, ny);
  float*** vpr = sf_floatalloc3(nz, nx, ny);
  float*** tmparray = sf_floatalloc3(nz0, nx0, ny0);
  
  sf_floatread(tmparray[0][0], nz0 * nx0 * ny0, file_mdl);
  expand3d(tmparray, vpz, az0, ax0, ay0, az, ax, ay);
  
  sf_floatread(tmparray[0][0], nz0 * nx0 * ny0, file_mdl);
  expand3d(tmparray, vpx, az0, ax0, ay0, az, ax, ay);
  
  sf_floatread(tmparray[0][0], nz0 * nx0 * ny0, file_mdl);
  expand3d(tmparray, vpy, az0, ax0, ay0, az, ax, ay);
  
  sf_floatread(tmparray[0][0], nz0 * nx0 * ny0, file_mdl);
  expand3d(tmparray, vn1, az0, ax0, ay0, az, ax, ay);
  
  sf_floatread(tmparray[0][0], nz0 * nx0 * ny0, file_mdl);
  expand3d(tmparray, vn2, az0, ax0, ay0, az, ax, ay);
  
  sf_floatread(tmparray[0][0], nz0 * nx0 * ny0, file_mdl);
  expand3d(tmparray, vn3, az0, ax0, ay0, az, ax, ay);

  sf_floatread(tmparray[0][0], nz0 * nx0 * ny0, file_mdl);
  expand3d(tmparray, vpr, az0, ax0, ay0, az, ax, ay);
  
  free(**tmparray);
  free(*tmparray);
  free(tmparray);
  
  for (int iy = 0; iy < ny; iy++) {
    for (int ix = 0; ix < nx; ix++) {
      for (int iz = 0; iz < nz; iz++) {
        vpz[iy][ix][iz] *= vpz[iy][ix][iz]; // vpz
        vpx[iy][ix][iz] *= vpx[iy][ix][iz]; // vpx
        vpy[iy][ix][iz] *= vpy[iy][ix][iz]; // vpy
        vn1[iy][ix][iz] *= vn1[iy][ix][iz]; // vn1
        vn2[iy][ix][iz] *= vn2[iy][ix][iz]; // vn2
        vn3[iy][ix][iz] *= vn3[iy][ix][iz]; // vn3
        vpr[iy][ix][iz] *= vpr[iy][ix][iz] * dt * dt; // reference velocity
      }
    }
  }
  float*** vcomposite = vpr;

  
  float* kz = sf_floatalloc(nkz);
  float* kx = sf_floatalloc(nkx);
  float* ky = sf_floatalloc(nky);
  for (int iky = 0; iky < nky; ++iky) {
    float ky_ = oky + iky * dky;
    if (iky >= nky / 2) ky_ = (iky - nky) * dky;
    ky[iky] = powf(2 * SF_PI * ky_, 2);
  }
  
  for (int ikx = 0; ikx < nkx; ++ikx) {
    float kx_ = okx + ikx * dkx;
    if (ikx >= nkx / 2) kx_ = (ikx - nkx) * dkx;
    kx[ikx] = powf(2 * SF_PI * kx_, 2);
  }
  
  for (int ikz = 0; ikz < nkz; ++ikz) {
    float kz_ = okz + ikz * dkz;
    kz[ikz] = powf(2 * SF_PI * kz_, 2);
  }
  
  // lowrank decomposition
  size_t m = nz * nx * ny;
  size_t n = nkz * nkx * nky;
  void* ort_lrparam = lrdecomp_new(m, n);
  lrdecomp_init(ort_lrparam, seed, npk, eps, dt, nkz, nkx, nky, kz, kx, ky, vpz[0][0], vpx[0][0],
                vpy[0][0], vn1[0][0], vn2[0][0], vn3[0][0]);
  // lrmat* ort_lrmat = lrdecomp_compute(ort_lrparam, m, n, seed, npk, eps);
  lrmat* ort_lrmat = lrdecomp_compute(ort_lrparam);
  int nrank = ort_lrmat->nrank;
  float** lft = sf_floatalloc2(m, nrank);
  float** rht = sf_floatalloc2(n, nrank);
  memcpy(lft[0], ort_lrmat->lft_data, sizeof(float) * m * nrank);
  memcpy(rht[0], ort_lrmat->rht_data, sizeof(float) * n * nrank);
  lrdecomp_delete(ort_lrparam, ort_lrmat);
  
  // wave propagation with decomposed matrices
  sf_oaxa(file_dat, ar, 1);
  sf_oaxa(file_dat, at, 2);
  sf_oaxa(file_dat, as, 3);
  sf_fileflush(file_dat, NULL);
  
  if (snap) {
    sf_oaxa(file_wfl, az0, 1);
    sf_oaxa(file_wfl, ax0, 2);
    sf_oaxa(file_wfl, ay0, 3);
    sf_oaxa(file_wfl, at, 4);
    sf_oaxa(file_wfl, as, 5);
    sf_fileflush(file_wfl, NULL);
  }
  
  src3d = pt3dalloc2(1, ns);
  rec3d = pt3dalloc2(nr, ns);
  pt3dread2(file_src, src3d, 1, ns, 3);
  pt3dread2(file_rec, rec3d, nr, ns, 3);
  cssinc = (scoef3d*)malloc(ns * sizeof(*cssinc));
  crsinc = (scoef3d*)malloc(ns * sizeof(*crsinc));
  for (int is = 0; is < ns; is++) {
    cssinc[is] = sinc3d_make(1, src3d[is], az, ax, ay);
    crsinc[is] = sinc3d_make(nr, rec3d[is], az, ax, ay);
  }
  
  ws = sf_floatalloc2(ns, nt);
  sf_floatread(ws[0], ns * nt, file_inp);
  wr = sf_floatalloc2(nr, nt);
  u0 = sf_floatalloc3(nz, nx, ny);
  u1 = sf_floatalloc3(nz, nx, ny);
  
  float* rwavem = (float*)fftwf_malloc(nzpad * nxpad * nypad * sizeof(float));
  fftwf_complex* cwave =
  (fftwf_complex*)fftwf_malloc(nkz * nkx * nky * sizeof(fftwf_complex));
  fftwf_complex* cwavem =
  (fftwf_complex*)fftwf_malloc(nkz * nkx * nky * sizeof(fftwf_complex));
  
  /* mpi context set-up */
  int s_start_idx;
  int ns_local = ns / nprocs;
  /* if (myrank < (ns % nprocs)) ns_local++; */
  if ((nprocs - myrank) <= (ns % nprocs)) ns_local++;
  MPI_Scan(&ns_local, &s_start_idx, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  s_start_idx -= ns_local;
  
  sf_warning("ns=%d ns_local=%d s_start=%d myrannk=%d nprocs=%d ", ns, ns_local,
             s_start_idx, myrank, nprocs);
  
  /* boundary conditions */
  float*** ucut = sf_floatalloc3(nz0, nx0, ny0);
  float* damp = damp_make(nb);
  if (myrank == 0) {
    for (int is = 0; is < ns; is++) {
      if (snap) {
        for (int it = 0; it < nt; it++) {
          sf_floatwrite(ucut[0][0], nz0 * nx0 * ny0, file_wfl);
        }
      }
      sf_floatwrite(wr[0], nr * nt, file_dat);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  
  // float wt = (dt * dt) / (nxpad * nzpad * nypad);
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
  
  for (int is = 0; is < ns_local; is++) {
    int s_idx = is + s_start_idx;
    if (verb) fprintf(stderr, "\nshot # %d\n", s_idx);
    off_t dat_disp = (off_t)(s_idx)*nr * nt * sizeof(float);
    off_t wfl_disp = (off_t)(s_idx)*nz0 * nx0 * ny0 * sizeof(float);
    sf_seek(file_dat, dat_disp, SEEK_SET);
    if (snap) sf_seek(file_wfl, wfl_disp, SEEK_SET);
    
    float*** ptrtmp = NULL;
    memset(u0[0][0], 0, sizeof(float) * nz * nx * ny);
    memset(u1[0][0], 0, sizeof(float) * nz * nx * ny);
    
    for (int it = 0; it < nt; it++) {
      if (verb) sf_warning("it = %d;", it);
      double tic = omp_get_wtime();
      
      /* apply absorbing boundary condition: E \times u@n-1 */
      lr_fft_stepforward(u0, u1, rwavem, cwave, cwavem, lft, rht, forward_plan,
                         inverse_plan, nz, nx, ny, nzpad, nxpad, nypad, nkz,
                         nkx, nky, nrank, wt, false);
      
      // sinc3d_inject1(u0, ws[it][s_idx], cssinc[s_idx]);
      sinc3d_inject1_with_vv(u0, ws[it][s_idx], cssinc[s_idx], vcomposite);
      
      /* apply absorbing boundary condition: E \times u@n+1 */
      damp3d_apply(u1, damp, nz, nx, ny, nb);
      damp3d_apply(u0, damp, nz, nx, ny, nb);
      
      /* loop over pointers */
      ptrtmp = u0;
      u0 = u1;
      u1 = ptrtmp;
      
      if (snap) {
        window3d(ucut, u0, az0, ax0, ay0, az, ax, ay);
        sf_floatwrite(ucut[0][0], nz0 * nx0 * ny0, file_wfl);
      }
      
      sinc3d_extract(u0, wr[it], crsinc[s_idx]);
      
      double toc = omp_get_wtime();
      if (verb) fprintf(stderr, " clock = %lf;", toc - tic);
    } /* END OF FORWARD TIME LOOP */
    
    sf_floatwrite(wr[0], nr * nt, file_dat);
  } // END OF SRC LOOP
  /* MPI_Barrier(MPI_COMM_WORLD); */
  double etime = MPI_Wtime();
  if (verb) sf_warning(" Elapsed time = %lf;", etime - stime);
  MPI_Finalize();
  return 0;
}
