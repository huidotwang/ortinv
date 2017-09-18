#include "misfit.h"

double
compute_misfit_function_adjoint_source_l2xcor(float** dcal, float** dobs, int nt, int nr)
{
  double misfit = 0.0;
  for (int it = 0; it < nt; it++) {
    for (int ir = 0; ir < nr; ir++) {
      misfit += powf(dobs[it][ir] * dcal[it][ir], 2);
    }
  }
  /* adjoint source */
#pragma omp parallel for schedule(dynamic, 1)
  for (int it = 0; it < nt; it++) {
    for (int ir = 0; ir < nr; ir++) {
      dcal[it][ir] *= -powf(dobs[it][ir], 2);
    }
  }
  return -misfit;
}

double
compute_misfit_function_adjoint_source_l2data(float** dcal, float** dobs,
                                              int nt, int nr)
{
  /* adjoint source */
#pragma omp parallel for schedule(dynamic, 1)
  for (int it = 0; it < nt; it++) {
    for (int ir = 0; ir < nr; ir++) {
      dcal[it][ir] -= dobs[it][ir];
    }
  }
  double misfit = 0.0;
  for (int it = 0; it < nt; it++) {
    for (int ir = 0; ir < nr; ir++) {
      misfit += powf(dcal[it][ir], 2);
    }
  }
  return misfit;
}

double
compute_misfit_function_adjoint_source_l2envelope(float** dcal, float** dobs,
                                                  int nt, int nr)
{
  float** dcal_t = sf_floatalloc2(nt, nr);
  float** dobs_t = sf_floatalloc2(nt, nr);
  float** Hdcal_t = sf_floatalloc2(nt, nr);
  float** Hdobs_t = sf_floatalloc2(nt, nr);
  float** dE2_t = sf_floatalloc2(nt, nr);
  float** dE2_Hdcal_t = Hdobs_t;
  float** H_dE2_Hdcal_t = dobs_t;
  double misfit = 0.0;
  /* transpose dcal, dobs */
#pragma omp parallel for schedule(dynamic, 1)
  for (int it = 0; it < nt; it++) {
    for (int ir = 0; ir < nr; ir++) {
      dcal_t[ir][it] = dcal[it][ir];
      dobs_t[ir][it] = dobs[it][ir];
    }
  }
  /* Hilbert transform */
  hilbert(Hdcal_t[0], dcal_t[0], nt, nr);
  hilbert(Hdobs_t[0], dobs_t[0], nt, nr);
#pragma omp parallel for schedule(dynamic, 1)
  for (int ir = 0; ir < nr; ir++) {
    for (int it = 0; it < nt; it++) {
      dE2_t[ir][it] = powf(dcal_t[ir][it], 2) + powf(Hdcal_t[ir][it], 2) -
      powf(dobs_t[ir][it], 2) - powf(Hdobs_t[ir][it], 2);
      dE2_Hdcal_t[ir][it] = Hdcal_t[ir][it] * dE2_t[ir][it];
    }
  }
  hilbert(H_dE2_Hdcal_t[0], dE2_Hdcal_t[0], nt, nr);
  /* adjoint source function */
#pragma omp parallel for schedule(dynamic, 1)
  for (int ir = 0; ir < nr; ir++) {
    for (int it = 0; it < nt; it++) {
      dcal[it][ir] =
      2.f * (dE2_t[ir][it] * dcal_t[ir][it] - H_dE2_Hdcal_t[ir][it]);
    }
  }
  /* misfit function */
  for (int ir = 0; ir < nr; ir++) {
    for (int it = 0; it < nt; it++) {
      misfit += powf(dE2_t[ir][it], 2);
    }
  }
  free(*dcal_t); free(dcal_t);
  free(*dobs_t); free(dobs_t);
  free(*Hdcal_t); free(Hdcal_t);
  free(*Hdobs_t); free(Hdobs_t);
  free(*dE2_t); free(dE2_t);
  return misfit;
}
