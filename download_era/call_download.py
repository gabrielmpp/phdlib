import subprocess
import numpy as np
import sys
import os
from pathlib import Path

if __name__ == '__main__':
    this_file_path = Path(os.path.dirname(os.path.abspath(__file__)))

    logs_path = '/home/users/gmpp/logs/'
    python = '/home/users/gmpp/miniconda2/envs/phd37/bin/python'
    # python = '/home/gab/software/miniconda3/envs/phd37/bin/python'
    # script_paths = ['/home/users/gmpp/phdlib/download_era/download_era_viwvn.py',]
    script_paths = [
                    #this_file_path / 'download_era_viwvn.py',
                    this_file_path / 'download_era_convective.py',
                    #this_file_path / 'download_era_tcwv.py'
    ]
    years = np.arange(1979, 2020, 1)
    jasmin = True  # also change config, paths and python
    if jasmin:

        for script_path in script_paths:
            for year in years:
                subprocess.call(['bsub',
                                 '-o', logs_path + '%J.out',
                                 '-e', logs_path + '%J.err',
                                 '-W', '24:00',
                                 # '-R','rusage[mem=5000]',
                                 # '-M','5000',
                                 # '-n','2',
                                 '-q', 'short-serial',
                                 python, str(script_path), str(year)])
    else:

        for script_path in script_paths:
            for year in years:
                print(f'Running year {year}')
                subprocess.call(['nohup',
                                 python, str(script_path), str(year)])
