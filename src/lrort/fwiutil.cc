#include "fwiutil.hh"
#include <cassert>

fwi_object_t*
fwi_init(sf_file _file_dat, sf_file _file_wav, sf_file _file_src,
         sf_file _file_rec, sf_axis _az0, sf_axis _ax0, sf_axis _ay0,
         sf_axis _as, sf_axis _ar, sf_axis _at, int _nb, int MPI_MASTER,
         char* localdatapath)
{
  fwi_object_t* p = (fwi_object_t*)malloc(sizeof(*p));
  p->file_dat = _file_dat;
  p->az0 = _az0;
  p->ax0 = _ax0;
  p->ay0 = _ay0;
  p->as = _as;
  p->ar = _ar;
  p->at = _at;
  p->nb = _nb;
  float dz = sf_d(_az0);
  float dx = sf_d(_ax0);
  float dy = sf_d(_ay0);
  int nz = sf_n(_az0) + 2 * _nb;
  int nx = sf_n(_ax0) + 2 * _nb;
  int ny = sf_n(_ay0) + 2 * _nb;
  float oz = sf_o(_az0) - dz * _nb;
  float ox = sf_o(_ax0) - dx * _nb;
  float oy = sf_o(_ay0) - dy * _nb;
  p->az = sf_maxa(nz, oz, dz);
  p->ax = sf_maxa(nx, ox, dx);
  p->ay = sf_maxa(ny, oy, dy);
  int nzpad = kiss_fft_next_fast_size(((nz + 1) >> 1) << 1);
  int nxpad = kiss_fft_next_fast_size(nx);
  int nypad = kiss_fft_next_fast_size(ny);
  p->azpad = sf_maxa(nzpad, oz, dz);
  p->axpad = sf_maxa(nxpad, ox, dx);
  p->aypad = sf_maxa(nypad, oy, dy);
  int nkz = nzpad / 2 + 1;
  int nkx = nxpad;
  int nky = nypad;
  float okz = 0.f;
  float okx = 0.f;
  float oky = 0.f;
  float dkz = 1.f / (nzpad * dz);
  float dkx = 1.f / (nxpad * dx);
  float dky = 1.f / (nypad * dy);
  p->akz = sf_maxa(nkz, okz, dkz);
  p->akx = sf_maxa(nkx, okx, dkx);
  p->aky = sf_maxa(nky, oky, dky);

  p->kk[0] = sf_floatalloc(nkz);
  p->kk[1] = sf_floatalloc(nkx);
  p->kk[2] = sf_floatalloc(nky);

  for (int iky = 0; iky < nky; ++iky) {
    p->kk[2][iky] = oky + iky * dky;
    if (iky >= nky / 2) p->kk[2][iky] = (iky - nky) * dky;
    p->kk[2][iky] *= 2 * SF_PI;
    p->kk[2][iky] *= p->kk[2][iky];
  }

  for (int ikx = 0; ikx < nkx; ++ikx) {
    p->kk[1][ikx] = okx + ikx * dkx;
    if (ikx >= nkx / 2) p->kk[1][ikx] = (ikx - nkx) * dkx;
    p->kk[1][ikx] *= 2 * SF_PI;
    p->kk[1][ikx] *= p->kk[1][ikx];
  }
  for (int ikz = 0; ikz < nkz; ++ikz) {
    p->kk[0][ikz] = okz + ikz * dkz;
    p->kk[0][ikz] *= 2 * SF_PI;
    p->kk[0][ikz] *= p->kk[0][ikz];
  }

  /* pre-compute sinc interpolation coefficients */
  int ns = sf_n(p->as);
  int nr = sf_n(p->ar);
  pt3d** src3d = pt3dalloc2(1, ns);  /* source position */
  pt3d** rec3d = pt3dalloc2(nr, ns); /*receiver position*/
  pt3dread2(_file_src, src3d, 1, ns, 3);
  pt3dread2(_file_rec, rec3d, nr, ns, 3);
  p->cssinc = (scoef3**)malloc(ns * sizeof(*(p->cssinc)));
  p->crsinc = (scoef3**)malloc(ns * sizeof(*(p->crsinc)));
  for (int is = 0; is < ns; is++) {
    p->cssinc[is] = sinc3d_make(1, src3d[is], p->az, p->ax, p->ay);
    p->crsinc[is] = sinc3d_make(nr, rec3d[is], p->az, p->ax, p->ay);
  }
  FREE2D(src3d);
  FREE2D(rec3d);

  /* source functions */
  int nt = sf_n(_at);
  p->ws = sf_floatalloc2(ns, nt);
  sf_floatread(p->ws[0], ns * nt, _file_wav);

  /* pre-compute boudary damping profile */
  p->damp = damp_make(p->nb);

  /* allocate referece velocity array */
  p->vr = sf_floatalloc3(nz, nx, ny);

  /* MPI utility */
  p->mmpi = (mpi_context_t*)malloc(sizeof(*(p->mmpi)));
  MPI_Comm_size(MPI_COMM_WORLD, &p->mmpi->nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &p->mmpi->myrank);
  p->mmpi->MASTER = MPI_MASTER;
  p->mmpi->ns_local = ns / p->mmpi->nprocs;
  /* if (p->mmpi->myrank < (ns % p->mmpi->nprocs)) p->mmpi->ns_local++; */
  if ((p->mmpi->nprocs - p->mmpi->myrank) <= (ns % p->mmpi->nprocs))
    p->mmpi->ns_local++;
  MPI_Scan(&p->mmpi->ns_local, &p->mmpi->s_start_idx, 1, MPI_INT, MPI_SUM,
           MPI_COMM_WORLD);
  p->mmpi->s_start_idx -= p->mmpi->ns_local;
  sf_warning("ns=%d ns_local=%d s_start=%d myrannk=%d nprocs=%d ", ns,
             p->mmpi->ns_local, p->mmpi->s_start_idx, p->mmpi->myrank,
             p->mmpi->nprocs);

  /* local disk utility */
  p->my_file_pack = (file_pack*)malloc(sizeof(*p->my_file_pack));
  create_swfl_file(p->my_file_pack, p->mmpi->myrank, localdatapath);

  return p;
}

void
fwi_finalize(fwi_object_t* p)
{
  FREE3D(p->vr);
  free(p->kk[0]);
  free(p->kk[1]);
  free(p->kk[2]);
  free(p->ws);
  free(p->damp);
  delete_swfl_file(p->my_file_pack);
}

static void
forward_modeling(float** dat_cal, float** ws, float*** vcomposite,
                 float* rwavem, fftwf_complex* cwave, fftwf_complex* cwavem,
                 float** lft, float** rht, fftwf_plan forward_plan,
                 fftwf_plan inverse_plan, int nz, int nx, int ny, int nzpad,
                 int nxpad, int nypad, int nkz, int nkx, int nky, int nb,
                 int nrank, int s_idx, int nt, float dt, float wt, float* damp,
                 scoef3d* cssinc, scoef3d* crsinc, FILE* fh_swfl, float**** us)
/* compute state variable and calculated data */
{
  float*** u0 = sf_floatalloc3(nz, nx, ny);
  float*** u1 = sf_floatalloc3(nz, nx, ny);
  float*** uptr = NULL;
  memset(u0[0][0], 0, sizeof(float) * ny * nx * nz);
  memset(u1[0][0], 0, sizeof(float) * ny * nx * nz);
  // int nbell = 10;
  // float*** bell3d = bell3d_init(nbell);
  for (int it = 0; it < nt; it++) {
    /* sf_warning("it=%d;",it+1); */
#if 0
    memcpy(us[it][0][0], u0[0][0], sizeof(float) * nz * nx * ny);
#endif
    lr_fft_stepforward(u0, u1, rwavem, cwave, cwavem, lft, rht, forward_plan,
                       inverse_plan, nz, nx, ny, nzpad, nxpad, nypad, nkz, nkx,
                       nky, nrank, wt, false);
    // sinc3d_inject1(u0, ws[it][s_idx], cssinc[s_idx]);
    sinc3d_inject1_with_vv(u0, ws[it][s_idx], cssinc[s_idx], vcomposite);
    damp3d_apply(u1, damp, nz, nx, ny, nb);
    damp3d_apply(u0, damp, nz, nx, ny, nb);
    SWAP(u0, u1, uptr);
    sinc3d_extract(u0, dat_cal[it], crsinc[s_idx]);
#if 0
#pragma omp parallel for schedule(dynamic, 1)
    for (int iy = 0; iy < ny; iy++) {
      for (int ix = 0; ix < nx; ix++) {
        for (int iz = 0; iz < nz; iz++) {
          us[it][iy][ix][iz] =
            (u1[iy][ix][iz] + us[it][iy][ix][iz] - 2.0 * u0[iy][ix][iz]) /
            (dt * dt);
        }
      }
    }
#endif
    memcpy(us[it][0][0], u0[0][0], sizeof(float) * nz * nx * ny);

    // fwrite(u0[0][0], sizeof(float), (size_t)(nz)*nx * ny, fh_swfl);
    // bell3d_apply(us[it], bell3d, nbell, cssinc[s_idx]);
  }
  FREE3D(u0);
  FREE3D(u1);
  // FREE3D(bell3d);
  return;
}

static void
backward_modeling(float**** gradients, float** adj_src, float*** vp[],
                  float*** vn[], float*** vr, float*** vcomposite, float* kk[],
                  float* rwavem, fftwf_complex* cwave, fftwf_complex* cwavem,
                  float** lft, float** rht, fftwf_plan forward_plan,
                  fftwf_plan inverse_plan, int nrank, int nb, int nz0, int nx0,
                  int ny0, int nz, int nx, int ny, int nzpad, int nxpad,
                  int nypad, int nkz, int nkx, int nky, int s_idx, int nt,
                  float wt, float* damp, scoef3d* crsinc, FILE* fh_swfl,
                  float**** us)
{
  float*** u0 = sf_floatalloc3(nz, nx, ny);
  float*** u1 = sf_floatalloc3(nz, nx, ny);
  float*** uptr = NULL;
  memset(u0[0][0], 0, sizeof(float) * ny * nx * nz);
  memset(u1[0][0], 0, sizeof(float) * ny * nx * nz);
  for (int it = nt - 1; it >= 0; it--) {
    /* sf_warning("it=%d;",it+1); */
    damp3d_apply(u0, damp, nz, nx, ny, nb);
    lr_fft_stepforward(u0, u1, rwavem, cwave, cwavem, lft, rht, forward_plan,
                       inverse_plan, nz, nx, ny, nzpad, nxpad, nypad, nkz, nkx,
                       nky, nrank, wt, true);
    // sinc3d_inject(u0, adj_src[it], crsinc[s_idx]);
    sinc3d_inject_with_vv(u0, adj_src[it], crsinc[s_idx], vcomposite);
    damp3d_apply(u0, damp, nz, nx, ny, nb);
    SWAP(u0, u1, uptr);

    /* fseeko(fh_swfl, (off_t)(it)*nz * nx * ny * sizeof(float), SEEK_SET);
    fread(us[0][0], sizeof(float), (size_t)(nz)*nx * ny, fh_swfl); */

    compute_gradients(us[it], u0, gradients, rwavem, cwave, cwavem,
                      forward_plan, inverse_plan, vp, vn, vr, kk, nz, nx, ny,
                      nzpad, nxpad, nypad, nkz, nkx, nky, wt);
  }
  FREE3D(u0);
  FREE3D(u1);
  // FREE3D(us);
  return;
}

static double
fwi_fdf_local(float**** vel2, float**** gradients, fwi_object_t* mfwi,
              mpi_context_t* mmpi)
{
  int nz0 = sf_n(mfwi->az0);
  int nx0 = sf_n(mfwi->ax0);
  int ny0 = sf_n(mfwi->ay0);
  int nz = sf_n(mfwi->az);
  int nx = sf_n(mfwi->ax);
  int ny = sf_n(mfwi->ay);
  int nzpad = sf_n(mfwi->azpad);
  int nxpad = sf_n(mfwi->axpad);
  int nypad = sf_n(mfwi->aypad);
  int nkz = sf_n(mfwi->akz);
  int nkx = sf_n(mfwi->akx);
  int nky = sf_n(mfwi->aky);
  int nb = mfwi->nb;
  int nr = sf_n(mfwi->ar);
  int nt = sf_n(mfwi->at);
  float dt = sf_d(mfwi->at);
  float** kk = mfwi->kk;
  float* damp = mfwi->damp;
  float** ws = mfwi->ws;
  scoef3d* cssinc = mfwi->cssinc;
  scoef3d* crsinc = mfwi->crsinc;
  FILE* fh_swfl = mfwi->my_file_pack->fh_swfl;
  float**** vp = &vel2[0];
  float**** vn = &vel2[3];
  float*** vr = mfwi->vr;

  float*** vcomposite = sf_floatalloc3(nz, nx, ny);
  for (int iy = 0; iy < ny; iy++) {
    for (int ix = 0; ix < nx; ix++) {
      for (int iz = 0; iz < nz; iz++) {
        vcomposite[iy][ix][iz] = dt * dt / vr[iy][ix][iz];
      }
    }
  }

  // lowrank_decomposition();
  int seed = mfwi->seed;
  int npk = mfwi->npk;
  float eps = mfwi->eps;
  size_t m = nz * nx * ny;
  size_t n = nkz * nkx * nky;
  void* ort_lrparam = lrdecomp_new(m, n);
  lrdecomp_init(ort_lrparam, seed, npk, eps, dt, nkz, nkx, nky, kk[0], kk[1],
                kk[2], vp[0][0][0], vp[1][0][0], vp[2][0][0], vn[0][0][0],
                vn[1][0][0], vn[2][0][0]);
  // lrmat* ort_lrmat = lrdecomp_compute(ort_lrparam, m, n, seed, npk, eps);
  lrmat* ort_lrmat = lrdecomp_compute(ort_lrparam);
  int nrank = ort_lrmat->nrank;
  float** lft = sf_floatalloc2(m, nrank);
  float** rht = sf_floatalloc2(n, nrank);
  memcpy(lft[0], ort_lrmat->lft_data, sizeof(float) * m * nrank);
  memcpy(rht[0], ort_lrmat->rht_data, sizeof(float) * n * nrank);
  lrdecomp_delete(ort_lrparam, ort_lrmat);

  float**** us = sf_floatalloc4(nz, nx, ny, nt);
  // lowrank_wave_propagation();
  float** dobs = sf_floatalloc2(nr, nt);
  float** dcal = sf_floatalloc2(nr, nt);
  float* rwavem = (float*)fftwf_malloc(nzpad * nxpad * nypad * sizeof(float));
  fftwf_complex* cwave =
    (fftwf_complex*)fftwf_malloc(nkz * nkx * nky * sizeof(fftwf_complex));
  fftwf_complex* cwavem =
    (fftwf_complex*)fftwf_malloc(nkz * nkx * nky * sizeof(fftwf_complex));

  float wt = 1.f / (nxpad * nzpad * nypad);
  fftwf_init_threads();
  fftwf_plan_with_nthreads(omp_get_max_threads());
  fftwf_plan forward_plan =
    fftwf_plan_dft_r2c_3d(nypad, nxpad, nzpad, rwavem, cwave, FFTW_MEASURE);
  fftwf_plan_with_nthreads(omp_get_max_threads());
  fftwf_plan inverse_plan =
    fftwf_plan_dft_c2r_3d(nypad, nxpad, nzpad, cwavem, rwavem, FFTW_MEASURE);

  double local_fcost = 0.0;
  for (int is = 0; is < mmpi->ns_local; is++) {
    int s_idx = is + mmpi->s_start_idx;

    /* compute state variable and calculated data */
    forward_modeling(dcal, ws, vcomposite, rwavem, cwave, cwavem, lft, rht,
                     forward_plan, inverse_plan, nz, nx, ny, nzpad, nxpad,
                     nypad, nkz, nkx, nky, nb, nrank, s_idx, nt, dt, wt, damp,
                     cssinc, crsinc, fh_swfl, us);

    /* compute misfit function and adjoint source */
    off_t dat_disp = (off_t)(s_idx)*nr * nt * sizeof(float);
    sf_seek(mfwi->file_dat, dat_disp, SEEK_SET);
    sf_floatread(dobs[0], nr * nt, mfwi->file_dat);
    local_fcost += compute_misfit_function_adjoint_source(dcal, dobs, nt, nr);

    /* compute gradient */
    backward_modeling(gradients, dcal, vp, vn, vr, vcomposite, kk, rwavem,
                      cwave, cwavem, lft, rht, forward_plan, inverse_plan,
                      nrank, nb, nz0, nx0, ny0, nz, nx, ny, nzpad, nxpad, nypad,
                      nkz, nkx, nky, s_idx, nt, wt, damp, crsinc, fh_swfl, us);
  }
  FREE3D(vcomposite);
  FREE4D(us);
  FREE2D(dobs);
  FREE2D(dcal);
  FREE2D(lft);
  FREE2D(rht);
  free(rwavem);
  free(cwave);
  free(cwavem);
  fftwf_destroy_plan(forward_plan);
  fftwf_destroy_plan(inverse_plan);
  return local_fcost;
}

static void
dbl2flt(bool adj, double* a, float**** b, sf_axis az0, sf_axis ax0, sf_axis ay0,
        sf_axis az, sf_axis ax, sf_axis ay)
{
  int nz0 = sf_n(az0);
  int nx0 = sf_n(ax0);
  int ny0 = sf_n(ay0);
  size_t nvars0 = nz0 * nx0 * ny0 * 6;
  float**** tmparray = sf_floatalloc4(nz0, nx0, ny0, 6);

  if (adj) {
    /* window operation */
    for (int i = 0; i < 6; i++)
      window3d(tmparray[i], b[i], az0, ax0, ay0, az, ax, ay);
    /* POKE operation */
    for (size_t i = 0; i < nvars0; i++) a[i] = *(tmparray[0][0][0] + i);
  } else {
    /* PEEK operation */
    for (size_t i = 0; i < nvars0; i++) *(tmparray[0][0][0] + i) = a[i];
    /* pad operation */
    for (int i = 0; i < 6; i++)
      expand3d(tmparray[i], b[i], az0, ax0, ay0, az, ax, ay);
  }
  FREE4D(tmparray);
  return;
}

void
fwi_fdf(double* x, double* fcost, double* g, fwi_object_t* mfwi)
{
  mpi_context_t* mmpi = mfwi->mmpi;
  int nz0 = sf_n(mfwi->az0);
  int nx0 = sf_n(mfwi->ax0);
  int ny0 = sf_n(mfwi->ay0);
  int nz = sf_n(mfwi->az);
  int nx = sf_n(mfwi->ax);
  int ny = sf_n(mfwi->ay);
  size_t nvars0 = (size_t)(nz0)*nx0 * ny0 * 6;
  size_t nvars = (size_t)(nz)*nx * ny * 6;

  float**** rsf_flt_vel = sf_floatalloc4(nz, nx, ny, 6);
  dbl2flt(false, x, rsf_flt_vel, mfwi->az0, mfwi->ax0, mfwi->ay0, mfwi->az,
          mfwi->ax, mfwi->ay);

  float**** rsf_flt_grd = sf_floatalloc4(nz, nx, ny, 6);
  memset(rsf_flt_grd[0][0][0], 0, sizeof(float) * nvars);

  /* forward modeling on local nodes */
  *fcost = fwi_fdf_local(rsf_flt_vel, rsf_flt_grd, mfwi, mmpi);
  *fcost *= 0.5;

  FREE4D(rsf_flt_vel);
  dbl2flt(true, g, rsf_flt_grd, mfwi->az0, mfwi->ax0, mfwi->ay0, mfwi->az,
          mfwi->ax, mfwi->ay);
  FREE4D(rsf_flt_grd);

  // descent direction is minus the gradient
  for (size_t i = 0; i < nvars0; i++) g[i] *= -1;

#if 0
  // scaling factor to make norm of gradient comparable to norm of unknowns
  float dt = sf_d(mfwi->at);
  double scalar = dt * dt;
  for (size_t i = 0; i < nvars0; i++) g[i] *= scalar;
#endif
  if (mmpi->myrank == mmpi->MASTER) {
    MPI_Reduce(MPI_IN_PLACE, fcost, 1, MPI_DOUBLE, MPI_SUM, mmpi->MASTER,
               MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, g, (int)nvars0, MPI_DOUBLE, MPI_SUM, mmpi->MASTER,
               MPI_COMM_WORLD);
  } else {
    MPI_Reduce(fcost, fcost, 1, MPI_DOUBLE, MPI_SUM, mmpi->MASTER,
               MPI_COMM_WORLD);
    MPI_Reduce(g, g, (int)nvars0, MPI_DOUBLE, MPI_SUM, mmpi->MASTER,
               MPI_COMM_WORLD);
  }
  return;
}
