#ifndef _GAUSSIAN_H
#define _GAUSSIAN_H
#include <stdlib.h>
#include "rsf.h"

typedef struct _gaussian gaussian_t;
struct _gausian
{
  size_t idx_x[9];
  size_t idx_y[9];
  size_t idx_z[9];
  float weit[9][9][9];
}

gaussian_t*
gaussian_init(int npt, pt2d* pts, sf_axis az, sf_axis ax, sf_axis ay)
{
  float oz = sf_o(az);
  float ox = sf_o(ax);
  float oy = sf_o(ay);
  float dz = sf_d(az);
  float dx = sf_d(ax);
  float dy = sf_d(ay);
  float variance_z = 2. * (4. * 4.);
  float variance_x = 2. * (4. * 4.);
  float variance_y = 2. * (4. * 4.);
  gaussian_t* gscoef = (gaussian_t*)malloc(sizeof(*pts) * npt);
  for (int ipt = 0; ipt < npt; ipt++) {
    float pt_z = pts[ipt].z;
    float pt_x = pts[ipt].x;
    float pt_y = pts[ipt].y;
    int pt_idx_z = floor((pt_z - oz) / dz);
    int pt_idx_x = floor((pt_x - ox) / dx);
    int pt_idx_y = floor((pt_y - oy) / dy);
    float pt_mod_z = pts[ipt].z - (pt_idx_z * dz + oz);
    float pt_mod_x = pts[ipt].x - (pt_idx_x * dx + ox);
    float pt_mod_y = pts[ipt].y - (pt_idx_y * dy + oy);
    for (int i = 0; i < 9; i++) {
      gscoef[ipt].idx_z[i] = pt_idx_z - 4 + i;
      gscoef[ipt].idx_x[i] = pt_idx_x - 4 + i;
      gscoef[ipt].idx_y[i] = pt_idx_y - 4 + i;
    }
    for (int iy = 0; iy < 9; iy++) {
      float dist_y = pt_y - (gscoef[ipt].idx_y[i] * dy + oy);
      float gy = expf(-dist_y * dist_y / variance_y);
      for (int ix = 0; ix < 9; ix++) {
        float dist_x = pt_x - (gscoef[ipt].idx_x[i] * dx + ox);
        float gx = expf(-dist_x * dist_x / variance_x);
        for (int iz = 0; iz < 9; iz++) {
          float dist_z = pt_z - (gscoef[ipt].idx_z[i] * dz + oz);
          float gz = expf(-dist_z * dist_z / variance_z);
          gscoef[ipt].weit[iy][ix][iz] = gy * gx * gz;
        }
      }
    }
    
  }
}

#endif
