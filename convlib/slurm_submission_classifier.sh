#!/bin/bash
#SBATCH -p long-serial
#SBATCH -o /home/users/gmpp/logs/%J.out
#SBATCH -e /home/users/gmpp/logs/%J.err
#SBATCH -t 160:00:00


/home/users/gmpp/miniconda2/envs/phd37/bin/python /home/users/gmpp/phdscripts/phdlib/convlib/classifier.py $@