#!/bin/bash
#SBATCH -p short-serial
#SBATCH -o /home/users/gmpp/logs2/Current_%J.out
#SBATCH -e /home/users/gmpp/logs2/Current_%J.err
#SBATCH -t 24:00:00
echo Arguments
echo $timestep
echo $SETTLS_order
/home/users/gmpp/miniconda2/envs/phd37/bin/python /home/users/gmpp/phdscripts/LagrangianCoherence/LCS/LCS.py $timestep $timedim $SETTLS_order $subdomain $savepath $ftlepath