#include "localdisk_helper.h"

void
create_swfl_file(file_pack *my_file_pack, int myrank, char* localdatapath)
{
  char *dir_comm;
  dir_comm = localdatapath;
  /* dir_comm = getenv("LOCALDATAPATH"); */
  if (!dir_comm) dir_comm = getenv("DATAPATH");
  fprintf(stderr,"\n DATAPATH = %s\n",dir_comm);
  char swfl_file_name[128];
  char swfl_dir[128];
  strcat(dir_comm,"/tmpswfl/");
  fprintf(stderr,"\n DATAPATH = %s\n",dir_comm);
  mkdir(dir_comm,0755);
  snprintf(swfl_dir, 128, "%siexp-%d",dir_comm,myrank);
  strcat(swfl_dir,"/");
  mkdir(swfl_dir,0755);
  snprintf(swfl_file_name, 128, "%sswfl-iexp-%d.rsf@",swfl_dir,myrank);
  FILE *fh_swfl = fopen(swfl_file_name, "w+b");
  my_file_pack->fh_swfl = fh_swfl;
  strcpy(my_file_pack->swfl_file_name, swfl_file_name);
  strcpy(my_file_pack->swfl_dir, swfl_dir);
  strcpy(my_file_pack->dir_comm, dir_comm);
  return;
}

void
delete_swfl_file(file_pack *my_file_pack)
{
  fclose(my_file_pack->fh_swfl);
  remove(my_file_pack->swfl_file_name);
  rmdir(my_file_pack->swfl_dir);
  rmdir(my_file_pack->dir_comm);
}

