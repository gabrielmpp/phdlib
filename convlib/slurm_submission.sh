#!/bin/bash
#SBATCH -p par-single
#SBATCH -o /home/users/gmpp/logs/%J.out
#SBATCH -e /home/users/gmpp/logs/%J.err
#SBATCH -t 24:00:00


/home/users/gmpp/miniconda2/envs/phd37/bin/python /home/users/gmpp/phdscripts/LagrangianCoherence/LCS/LCS.py $@