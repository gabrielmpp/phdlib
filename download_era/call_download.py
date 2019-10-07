import subprocess
import numpy as np
import sys
if __name__ == '__main__':
    logs_path = '/home/users/gmpp/logs/'
    python = '/home/users/gmpp/miniconda2/envs/phd37/bin/python'
    script_paths = ['/home/users/gmpp/phdlib/download_era/download_era_temp.py']
    years = np.arange(1979,2010,1)
    for script_path in script_paths:
        for year in years:
            subprocess.call(['bsub',
            '-o',logs_path+'%J.out',
            '-e',logs_path+'%J.err',
            '-W','24:00',
            #'-R','rusage[mem=5000]',
            #'-M','5000',
            #'-n','2',
            '-q','short-serial',
             python,script_path,str(year)])
