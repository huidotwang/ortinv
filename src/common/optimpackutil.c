#include "optimpackutil.h"

void
printout_iterinfo_nlcg(FILE* fstream, opk_nlcg_t* opt, const opk_vector_t* x,
                       const opk_vector_t* g, double f, double f0)
{
  time_t curtime;
  char time_str[20];
  time(&curtime);
  strftime(time_str, 20, "%H:%M %D", localtime(&curtime));

  if (opk_get_nlcg_evaluations(opt) == 0) {
    fprintf(fstream, "%4s %8s %12s %12s %12s %12s %14s %12s %19s\n", "ITER",
            "NEVAL", "ALPHA", "BETA", "|G|", "|X|", "F", "F/F0", "CLOCK");
  }
  fprintf(fstream, "%4ld %8ld %12.2E %12.2E %12.3E %12.3E %14.5E %12.3E %20s\n",
          (long)opk_get_nlcg_iterations(opt),
          (long)opk_get_nlcg_evaluations(opt), opk_get_nlcg_step(opt),
          opk_get_nlcg_beta(opt), opk_vnorm2(g), opk_vnorm2(x), f, f / f0,
          time_str);
  fflush(fstream);
  return;
}

void
printout_iterinfo_lbfgs(FILE* fstream, opk_lbfgs_t* opt, const opk_vector_t* x,
                        const opk_vector_t* g, double f, double f0)
{
  time_t curtime;
  char time_str[20];
  time(&curtime);
  strftime(time_str, 20, "%H:%M %D", localtime(&curtime));

  if (opk_get_lbfgs_evaluations(opt) == 0) {
    fprintf(fstream, "%4s %8s %12s %12s %12s %14s %12s %19s\n", "ITER", "NEVAL",
            "STEP", "|G|", "|X|", "F", "F/F0", "CLOCK");
  }
  fprintf(fstream, "%4ld %8ld %12.2E %12.3E %12.3E %14.5E %12.3E %20s\n",
          (long)opk_get_lbfgs_iterations(opt),
          (long)opk_get_lbfgs_evaluations(opt), opk_get_lbfgs_step(opt),
          opk_vnorm2(g), opk_vnorm2(x), f, f / f0, time_str);
  fflush(fstream);
  return;
}

void
printout_iterinfo_vmlmn(FILE* fstream, opk_vmlmn_t* opt, const opk_vector_t* x,
                        const opk_vector_t* g, double f, double f0)
{
  time_t curtime;
  char time_str[20];
  time(&curtime);
  strftime(time_str, 20, "%H:%M %D", localtime(&curtime));

  if (opk_get_vmlmn_evaluations(opt) == 0) {
    fprintf(fstream, "%4s %8s %12s %12s %12s %14s %12s %19s\n", "ITER", "NEVAL",
            "STEP", "|G|", "|X|", "F", "F/F0", "CLOCK");
  }
  fprintf(fstream, "%4ld %8ld %12.2E %12.3E %12.3E %14.5E %12.3E %20s\n",
          (long)opk_get_vmlmn_iterations(opt),
          (long)opk_get_vmlmn_evaluations(opt), opk_get_vmlmn_step(opt),
          opk_vnorm2(g), opk_vnorm2(x), f, f / f0, time_str);
  fflush(fstream);
  return;
}
