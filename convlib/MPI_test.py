# MPI myname
#BSUB -cwd /home/user/work
#BSUB -q par-multi
#BSUB -n 50
#BSUB -R rusage[mem=50000]
#BSUB -W 48:00
#BSUB -o %J.lo
#BSUB -e %J.err

# Load any environment modules (needed for mpi_myname.exe)
module load libfftw/intel/3.2.2_mpi

# Submit the job using
mpirun.lotus ./mpi_myname.exe
