#ifndef _LOCALDISK_HELPER_H
#define _LOCALDISK_HELPER_H
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

typedef struct file_pack_t file_pack;
struct file_pack_t {
  char swfl_file_name[128];
  char swfl_dir[128];
  char dir_comm[512];
  FILE *fh_swfl;
};

void create_swfl_file(file_pack *my_file_pack, int myrank, char* localdatapath);
void delete_swfl_file(file_pack *my_file_pack);

#endif
