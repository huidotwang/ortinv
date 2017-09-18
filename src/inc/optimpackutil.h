#include "optimpack.h"
#include <stdlib.h>
#include <time.h>

void printout_iterinfo_nlcg(FILE* fstream, opk_nlcg_t* opt,
                            const opk_vector_t* x, const opk_vector_t* g,
                            double f, double f0);

void printout_iterinfo_lbfgs(FILE* fstream, opk_lbfgs_t* opt,
                             const opk_vector_t* x, const opk_vector_t* g,
                             double f, double f0);

void printout_iterinfo_vmlmn(FILE* fstream, opk_vmlmn_t* opt,
                             const opk_vector_t* x, const opk_vector_t* g,
                             double f, double f0);
