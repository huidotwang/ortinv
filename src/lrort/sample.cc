/* vertical orthorhombic phase-term sampling module */
#include "sample.hh"
#include <cstdarg>
#include <cstring>
#include "vecmatop.hh"

/*
 * implementation of lrdecomp_ort constructor
 */
void*
lrdecomp_new(size_t m, size_t n)
{
  lrpar_ort* ortlrpar = new lrpar_ort;
  ortlrpar->base = new lrpar_base;
  ortlrpar->base->kk = new float*[3];
  ortlrpar->base->m = m;
  ortlrpar->base->n = n;
  return ortlrpar;
}
// vpz, vpx, vpy, vn1, vn2, vn3
void
lrdecomp_init(void* lrpar, int seed_, int npk_, float eps_, float dt_, ...)
{
  lrpar_ort* ortlrpar = (lrpar_ort*)lrpar;
  ortlrpar->base->seed = seed_;
  ortlrpar->base->npk = npk_;
  ortlrpar->base->eps = eps_;
  ortlrpar->base->dt = dt_;
  va_list ap;
  va_start(ap, dt_);
  ortlrpar->nkz = va_arg(ap, int);
  ortlrpar->nkx = va_arg(ap, int);
  ortlrpar->nky = va_arg(ap, int);
  ortlrpar->base->kk[0] = va_arg(ap, float*);
  ortlrpar->base->kk[1] = va_arg(ap, float*);
  ortlrpar->base->kk[2] = va_arg(ap, float*);
  ortlrpar->vpz = va_arg(ap, float*);
  ortlrpar->vpx = va_arg(ap, float*);
  ortlrpar->vpy = va_arg(ap, float*);
  ortlrpar->vn1 = va_arg(ap, float*);
  ortlrpar->vn2 = va_arg(ap, float*);
  ortlrpar->vn3 = va_arg(ap, float*);
  va_end(ap);
  return;
}

lrmat*
lrdecomp_compute(void* param)
{
  lrpar_ort* ortlrpar = (lrpar_ort*)param;
  size_t m = ortlrpar->base->m;
  size_t n = ortlrpar->base->n;
  int seed = ortlrpar->base->seed;
  int npk = ortlrpar->base->npk;
  float eps = ortlrpar->base->eps;
  srand48(seed);
  vector<size_t> lidx, ridx;
  MatrixXf mid;
  lowrank(m, n, sample, eps, npk, lidx, ridx, mid, param);
  size_t n2 = mid.cols();
  size_t m2 = mid.rows();
  vector<size_t> midx(m), nidx(n);
  for (size_t i = 0; i < m; i++) midx[i] = i;
  for (size_t i = 0; i < n; i++) nidx[i] = i;

  MatrixXf lmat_(m, m2);
  sample(midx, lidx, lmat_, param);
  MatrixXf lmat(m, n2);
  lmat = lmat_ * mid;

  MatrixXf rmat(n2, n);
  sample(ridx, nidx, rmat, param);
  rmat.transposeInPlace();

  lrmat* p = new lrmat;
  p->nrank = n2;
  p->lft_data = new float[m * n2];
  p->rht_data = new float[n * n2];
  memcpy(p->lft_data, lmat.data(), sizeof(float) * m * n2);
  memcpy(p->rht_data, rmat.data(), sizeof(float) * n * n2);
  return p;
}

void
lrdecomp_delete(void* lr_, void* fltmat_)
{
  lrpar_ort* lr = (lrpar_ort*)lr_;
  lr->vpz = NULL;
  lr->vpx = NULL;
  lr->vpy = NULL;
  lr->vn1 = NULL;
  lr->vn2 = NULL;
  lr->vn3 = NULL;
  delete lr->base->kk;
  delete lr->base;
  delete lr;
  lrmat* fltmat = (lrmat*)fltmat_;
  delete fltmat->lft_data;
  delete fltmat->rht_data;
  delete fltmat;
}

/*
 * map 1d array index to 3d array index
 */
static inline void
index1dto3d(size_t i, int n1, int n2, int n3, int& i1, int& i2, int& i3)
{
  size_t ii = i;
  i1 = ii % n1;
  ii /= n1;
  i2 = ii % n2;
  ii /= n2;
  i3 = ii % n3;
}

int
sample(vector<size_t>& rows, vector<size_t>& cols, MatrixXf& res, void* ebuf)
{
  lrpar_ort* m_param = (lrpar_ort*)ebuf;
  float dt = m_param->base->dt;
  float* kz = m_param->base->kk[0];
  float* kx = m_param->base->kk[1];
  float* ky = m_param->base->kk[2];
  float* vpz = m_param->vpz;
  float* vpx = m_param->vpx;
  float* vpy = m_param->vpy;
  float* vn1 = m_param->vn1;
  float* vn2 = m_param->vn2;
  float* vn3 = m_param->vn3;
  int nkx = m_param->nkx;
  int nky = m_param->nky;
  int nkz = m_param->nkz;

  size_t nrow = rows.size();
  size_t ncol = cols.size();
  res.resize(nrow, ncol);
  res.setZero(nrow, ncol);
  for (size_t ir = 0; ir < nrow; ir++) {
    float vpz_ = vpz[rows[ir]]; // vpz^2
    float vpx_ = vpx[rows[ir]]; // vpx^2
    float vpy_ = vpy[rows[ir]]; // vpy^2
    float vn1_ = vn1[rows[ir]]; // vn1^2
    float vn2_ = vn2[rows[ir]]; // vn2^2
    float vn3_ = vn3[rows[ir]]; // vn3^2
    for (size_t ic = 0; ic < ncol; ic++) {
      int ikx, iky, ikz;
      index1dto3d(cols[ic], nkz, nkx, nky, ikz, ikx, iky);
      float kx_ = kx[ikx]; // kx^2
      float ky_ = ky[iky]; // ky^2
      float kz_ = kz[ikz]; // kz^2
      float a, b, c;
      a = b = c = 0.f;
      a -= vpx_ * kx_;
      a -= vpy_ * ky_;
      a -= vpz_ * kz_;
      b += vpz_ * (vpx_ - vn1_) * kx_ * kz_;
      b += vpz_ * (vpy_ - vn2_) * ky_ * kz_;
      b += vpx_ * (vpy_ - vn3_) * kx_ * ky_;
      c += vpz_ * vpx_ * (vn2_ + vn3_);
      c -= vpz_ * sqrtf(vpx_ * vn1_ * vn2_ * vn3_) * 2.f;
      c += vpz_ * vpy_ * vn1_;
      c -= vpz_ * vpx_ * vpy_;
      c *= kx_ * ky_ * kz_;
      /* solve x^3 + a x^2 + b x + c = 0 */
      /* Possible reference: https://www.e-education.psu.edu/png520/m11_p6.html
       * http://www.trans4mind.com/personal_development/mathematics/polynomials/cubicAlgebra.htm
       */
      float A = b - a * a / 3.;
      float B = (2 * a * a * a - 9 * a * b + 27 * c) / 27.;
      float D = A * A * A / 27. + B * B / 4.;
      float cubic_root;
      if (std::fpclassify(D) == FP_ZERO) {
        float M = cbrtf(-0.5f * B);
        float N = M;
        float root1 = M + N - a / 3.0f;
        float root23 = -0.5f * (M + N) - a / 3.0f;
        cubic_root = std::max(root1, root23);
      } else if (D > 0.0f) {
        float M = cbrtf(-0.5f * B + sqrtf(D));
        float N = cbrtf(-0.5f * B - sqrtf(D));
        cubic_root = M + N - a / 3.0;
      } else {
        float cos_angle = sqrtf(-27. * B * B / (4. * A * A * A));
        if (B > 0.) cos_angle *= -1; 
        float angle = acosf(cos_angle);
        float root1 = 2.f * sqrtf(-A / 3.f) * cosf(angle / 3.f);
        float root2 = 2.f * sqrtf(-A / 3.f) * cosf((angle + 2.f * M_PI) / 3.f);
        float root3 = 2.f * sqrtf(-A / 3.f) * cosf((angle + 4.f * M_PI) / 3.f);
        cubic_root = std::max(std::max(root1, root2), root3);
        cubic_root -= a / 3.f;
      }
      if (cubic_root < 0.f) cubic_root = 0.f;
      float psi = sqrtf(cubic_root);
      res(ir, ic) = 2.f * cosf(psi * dt) - 2.f;
    }
  }
  return 0;
}
