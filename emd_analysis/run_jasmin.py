import subprocess
logs_path = '/home/users/gmpp/logs/'
python = '/home/users/gmpp/miniconda2/envs/phd37/bin/python'
script_path = '/home/users/gmpp/phdscripts/phdlib/emd_analysis/run_emd.py'

subprocess.call(['bsub',
                 '-o', logs_path + '%J.out',
                 '-e', logs_path + '%J.err',
                 '-x',
                 '-q', 'par-multi',
                 '-n', '201',
                 python, script_path])
