#include <math.h>
#include "rsf.h"
#include "hilbertf.h"

#ifdef SF_HAS_ENVELOPE
#define compute_misfit_function_adjoint_source                                 \
  compute_misfit_function_adjoint_source_l2envelope
#elif SF_HAS_XCOR
#define compute_misfit_function_adjoint_source                                 \
  compute_misfit_function_adjoint_source_l2xcor
#else
#define compute_misfit_function_adjoint_source                                 \
  compute_misfit_function_adjoint_source_l2data
#endif

double compute_misfit_function_adjoint_source(float** dcal, float** dobs,
                                              int nt, int nr);
