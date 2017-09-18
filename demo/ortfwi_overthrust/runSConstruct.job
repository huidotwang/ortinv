#!/bin/bash -x
#SBATCH --job-name="run"
#SBATCH -o "scons.log.out"
#SBATCH -e "scons.log.err"
#SBATCH --partition=compute
#SBATCH --exclusive
#SBATCH --export=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=24
#SBATCH --nodelist=compute[136-159]
#SBATCH --depend=afterok:1470525

# #SBATCH --nodelist=compute[159]
# #SBATCH --nodelist=compute[093-102,122-127,130-135,167,174-175]
# #SBATCH --nodes=1
# #SBATCH --nodelist=compute[175]
# #SBATCH --depend=afterok:1439500

# Go to the directoy from which our job was launched
cd $SLURM_SUBMIT_DIR

# Run the job.
export LOCALDATAPATH=/localscratch
rm -rf /local/tmpswfl
scons -f SConstruct
rm -rf /local/tmpswfl

# You can also use the following format to set 
# --nodes            - # of nodes to use
# --ntasks-per-node  - ntasks = nodes*ntasks-per-node
# --ntasks           - total number of MPI tasks
# srun --nodes=$NODES --ntasks=$TASKS --ntasks-per-node=$TPN $EXE > output.$JOBID

