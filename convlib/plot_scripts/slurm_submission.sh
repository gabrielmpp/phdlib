#!/bin/bash
#SBATCH -p short-serial
#SBATCH -o /home/users/gmpp/logs/plot2.out
#SBATCH -e /home/users/gmpp/logs/plot2.err
#SBATCH -t 24:00:00
#SBATCH --mem 30000
#SBATCH --job-name=plot2

/home/users/gmpp/miniconda2/envs/phd37/bin/python /home/users/gmpp/phdscripts/phdlib/convlib/plot_scripts/plot_correlation_rainfall.py 1