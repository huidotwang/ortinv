/*
 * module to sample phase-term of acousti orthorhombic media
 * with symmetry planes coincide with Cartesian coordinate plane
 */
#ifndef _SAMPLE_HH
#define _SAMPLE_HH
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include "lrdecomp.hh"

#ifdef EIGEN_DEFAULT_DENSE_INDEX_TYPE
#undef EIGEN_DEFAULT_DENSE_INDEX_TYPE
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE std::size_t
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace Eigen;
using std::vector;

typedef struct lrpar_ort_t lrpar_ort;
struct lrpar_ort_t
{
  lrpar_base* base;
  float *vpz, *vpx, *vpy, *vn1, *vn2, *vn3;
  int nkx, nky, nkz;
};

int sample(vector<size_t>&, vector<size_t>&, MatrixXf&, void*);

#endif
