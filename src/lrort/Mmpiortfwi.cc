/* ORT fwi with separable modeling kernel and OptimPack lbfgs optimization
 * engine */
#include <mpi.h>
extern "C" {
#include <complex.h>
#include <fftw3.h>
#include <omp.h>
#include "rsf.h"
#include "optimpackutil.h"
}
#include "fwiutil.hh"

static void
write_solution(sf_file file_out, double* dbl_x, size_t n)
{
  float* flt_x = new float[n];
  for (size_t i = 0; i < n; i++) flt_x[i] = sqrt(dbl_x[i]);
  sf_floatwrite(flt_x, n, file_out);
  sf_fflush(file_out);
  delete flt_x;
  return;
}

static void
hard_threshold(double* flt_x, float lb, float ub, size_t n)
{
/* two-sided hard thresholding  */
#pragma omp parallel for schedule(dynamic, 1)
  for (size_t i = 0; i < n; i++) {
    flt_x[i] = flt_x[i] < ub ? flt_x[i] : ub;
    flt_x[i] = flt_x[i] > lb ? flt_x[i] : lb;
  }
}

static void
apply_mask(double* flt_g, float* mask, size_t n)
{
#pragma omp parallel for schedule(dynamic, 1)
  for (size_t i = 0; i < n; i++) {
    flt_g[i] *= mask[i];
  }
  return;
}

int
main(int argc, char* argv[])
{
  bool verb;
  int nb;
  int max_iter_num;
  float flt_lb, flt_ub;
  int seed;
  int npk;
  float eps;

  sf_file file_wav = NULL;
  sf_file file_mdl = NULL;
  sf_file file_dat = NULL;
  sf_file file_src = NULL;
  sf_file file_rec = NULL;
  sf_file file_out = NULL;
  sf_file file_log = NULL;
  sf_file file_msk = NULL;

  sf_axis az0 = NULL, ax0 = NULL, ay0 = NULL;
  sf_axis at = NULL, as = NULL, ar = NULL;

  int nprocs, myrank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  /* double stime = MPI_Wtime(); */

  sf_init(argc, argv);

  if (!sf_getbool("verb", &verb)) verb = false;
  if (!sf_getint("nb", &nb)) nb = 4;
  if (!sf_getint("seed", &seed)) seed = time(NULL);
  if (!sf_getfloat("eps", &eps)) eps = 1e-7;
  if (!sf_getint("npk", &npk)) npk = 20;
  if (!sf_getint("maxiternum", &max_iter_num)) max_iter_num = 1;
  if (!sf_getfloat("lb", &flt_lb)) flt_lb = FLT_MIN;
  if (!sf_getfloat("ub", &flt_ub)) flt_ub = FLT_MAX;
  double dbl_lb = flt_lb * flt_lb;
  double dbl_ub = flt_ub * flt_ub;

  file_wav = sf_input("input");
  file_mdl = sf_input("model");
  file_src = sf_input("sou");
  file_rec = sf_input("rec");
  file_dat = sf_input("dat");
  if (sf_getstring("msk") != NULL) file_msk = sf_input("msk");
  char* localdatapath = sf_getstring("localdatapath");
  sf_warning("localdatapath = %s ", localdatapath);

  az0 = sf_iaxa(file_mdl, 1);
  ax0 = sf_iaxa(file_mdl, 2);
  ay0 = sf_iaxa(file_mdl, 3);
  at = sf_iaxa(file_wav, 2);
  as = sf_iaxa(file_src, 2);
  ar = sf_iaxa(file_rec, 2);

  file_out = sf_output("output");
  file_log = sf_output("log");
  sf_oaxa(file_out, az0, 1);
  sf_oaxa(file_out, ax0, 2);
  sf_oaxa(file_out, ay0, 3);
  sf_putint(file_out, "n4", 6);
  sf_fileflush(file_out, NULL);
  sf_oaxa(file_log, az0, 1);
  sf_oaxa(file_log, ax0, 2);
  sf_oaxa(file_log, ay0, 3);
  sf_putint(file_log, "n4", 6);
  sf_putint(file_log, "n5", max_iter_num + 1);
  sf_fileflush(file_log, NULL);

  int MPI_MASTER = nprocs - 1;
  size_t nvars = sf_n(az0) * sf_n(ax0) * sf_n(ay0) * 6;
  if (myrank == MPI_MASTER) fprintf(stderr, "Problem size: %zu\n", nvars);
  float* mask = NULL;
  if (file_msk != NULL) {
    mask = sf_floatalloc(nvars);
    sf_floatread(mask, nvars, file_msk);
  }
  fwi_object_t* mfwi =
    fwi_init(file_dat, file_wav, file_src, file_rec, az0, ax0, ay0, as, ar, at,
             nb, MPI_MASTER, localdatapath);
  mfwi->seed = seed;
  mfwi->npk = npk;
  mfwi->eps = eps;
  float* vel = sf_floatalloc(nvars);
  sf_floatread(vel, nvars, file_mdl);
  for (size_t i = 0; i < nvars; i++) vel[i] *= vel[i];
  float*** vpr = sf_floatalloc3(sf_n(az0), sf_n(ax0), sf_n(ay0));
  sf_floatread(vpr[0][0], sf_n(az0) * sf_n(ax0) * sf_n(ay0), file_mdl);
  for (int iy = 0; iy < sf_n(ay0); iy++) {
    for (int ix = 0; ix < sf_n(ax0); ix++) {
      for (int iz = 0; iz < sf_n(az0); iz++) {
        vpr[iy][ix][iz] = 1.f / (vpr[iy][ix][iz] * vpr[iy][ix][iz]);
      }
    }
  }
  expand3d(vpr, mfwi->vr, mfwi->az0, mfwi->ax0, mfwi->ay0, mfwi->az, mfwi->ax,
           mfwi->ay);
  FREE3D(vpr);

  // float* fgrad = (float*) malloc(sizeof(float) * nvars);
  double fcost;
  double* x_data = NULL;
  double* g_data = NULL;

  /* optimization with optimpack */
  int max_num_memory = 16;
  unsigned int vmlmn_flags = OPK_EMULATE_BLMVM;
  opk_vspace_t* vspace = opk_new_simple_double_vector_space(nvars);
  opk_vector_t* opk_x = opk_vcreate(vspace);
  opk_vector_t* opk_g = opk_vcreate(vspace);
  opk_bound_t* bnd_lb = opk_new_bound(vspace, OPK_BOUND_SCALAR, &dbl_lb);
  opk_bound_t* bnd_ub = opk_new_bound(vspace, OPK_BOUND_SCALAR, &dbl_ub);
  opk_vmlmn_t* opt = opk_new_vmlmn_optimizer(vspace, max_num_memory, vmlmn_flags, bnd_lb, bnd_ub, NULL);
  opk_task_t task;
  x_data = opk_get_simple_double_vector_data(opk_x);
  g_data = opk_get_simple_double_vector_data(opk_g);

  for (size_t i = 0; i < nvars; i++) x_data[i] = vel[i];
  free(vel);
  double f0 = 1.0;
  if (myrank == MPI_MASTER) task = opk_start_vmlmn(opt, opk_x);
  MPI_Bcast(&task, 1, MPI_INT, MPI_MASTER, MPI_COMM_WORLD);

  FILE* fh = NULL;
  if (myrank == MPI_MASTER) fh = fopen("iterate.dat", "a");

  int iter_num = 0;
  while (iter_num < max_iter_num) {
    if (task == OPK_TASK_NEW_X) {
      if (myrank == MPI_MASTER) write_solution(file_log, x_data, nvars);
    } else if (task == OPK_TASK_COMPUTE_FG) {
      // hard_threshold(x_data, lb, ub, nvars);
      fwi_fdf(x_data, &fcost, g_data, mfwi);
      if (file_msk != NULL) apply_mask(g_data, mask, nvars);
      if (opk_get_vmlmn_evaluations(opt) == 0)  f0 = fcost;
    } else { // OPK_TASK_ERROR, OPK_TASK_FINAL_X, OPK_TASK_WARNING
      break;
    }
    if (myrank == MPI_MASTER) {
      printout_iterinfo_vmlmn(stderr, opt, opk_x, opk_g, (float)fcost,
                              (float)f0);
      printout_iterinfo_vmlmn(fh, opt, opk_x, opk_g, (float)fcost, (float)f0);
      // update task and x
      task = opk_iterate_vmlmn(opt, opk_x, fcost, opk_g);
      iter_num = opk_get_vmlmn_iterations(opt);
    }
    MPI_Bcast(&iter_num, 1, MPI_INT, MPI_MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&task, 1, MPI_INT, MPI_MASTER, MPI_COMM_WORLD);
    MPI_Bcast(x_data, (int)nvars, MPI_DOUBLE, MPI_MASTER, MPI_COMM_WORLD);
  }

  /* post-optimization output */
  if (myrank == MPI_MASTER) {
    printout_iterinfo_vmlmn(stderr, opt, opk_x, opk_g, (float)fcost, (float)f0);
    printout_iterinfo_vmlmn(fh, opt, opk_x, opk_g, (float)fcost, (float)f0);
    write_solution(file_log, x_data, nvars);
    write_solution(file_out, x_data, nvars);
    const char* reason;
    if (task == OPK_TASK_FINAL_X) {
      reason = "Convergence";
    } else if (task == OPK_TASK_WARNING) {
      reason = "*** WARNING ***";
    } else if (task == OPK_TASK_ERROR) {
      reason = "*** ERROR ***";
    } else {
      reason = "*** Maximum iteration number reached ***";
    }
    fprintf(stderr, " - %s (%d) \n", reason, task);
    fprintf(fh, " - %s (%d) \n", reason, task);
    fclose(fh);
  }
  fwi_finalize(mfwi);
  MPI_Finalize();
  return 0;
}
