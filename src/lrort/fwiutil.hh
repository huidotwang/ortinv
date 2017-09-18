#ifndef _FWIUTIL_HH
#define _FWIUTIL_HH
#include <mpi.h>
extern "C" {
#include "ts_kernel.h"
#include "abcutil.h"
#include "Grid.h"
#include "sinc.h"
#include "rsf.h"
#include "misfit.h"
#include "localdisk_helper.h"
#include "rsf_wrapper.h"
#include "FDStencil.h"
}
#include "sample.hh"

#define SWAP(var1, var2, tmp)                                                  \
  tmp = var1;                                                                  \
  var1 = var2;                                                                 \
  var2 = tmp;

#define FREE2D(array)                                                          \
  free(*array);                                                                \
  free(array);

#define FREE3D(array)                                                          \
  free(**array);                                                               \
  free(*array);                                                                \
  free(array);

#define FREE4D(array)                                                          \
  free(***array);                                                              \
  free(**array);                                                               \
  free(*array);                                                                \
  free(array);

typedef struct _mpi_context mpi_context_t;
struct _mpi_context
{
  int MASTER;
  int nprocs;
  int myrank;
  int ns_local;
  int s_start_idx;
  MPI_Status m_status;
};

typedef struct _fwi_object fwi_object_t;
struct _fwi_object
{
  float** ws;
  float* damp;
  float* kk[3];
  float*** vr;
  sf_axis az0, ax0, ay0;
  sf_axis az, ax, ay;
  sf_axis azpad, axpad, aypad;
  sf_axis akz, akx, aky;
  sf_axis as;
  sf_axis ar;
  sf_axis at;
  sf_file file_dat;
  scoef3d* cssinc;
  scoef3d* crsinc;
  file_pack* my_file_pack;
  mpi_context_t* mmpi;
  int nop;
  int nb;
  int seed;
  int npk;
  float eps;
};

fwi_object_t* fwi_init(sf_file _file_dat, sf_file _file_wav, sf_file _file_src,
                       sf_file _file_rec, sf_axis _az0, sf_axis _ax0, sf_axis _ay0, sf_axis _as,
                       sf_axis _ar, sf_axis _at, int _nb, int MPI_MASTER, char* localdatapath);

void fwi_fdf(double* x, double* fcost, double* fgrad, fwi_object_t* mfwi);

void fwi_finalize(fwi_object_t* p);

#endif
