import subprocess
logs_path = '/home/users/gmpp/logs/'
python = '/home/users/gmpp/miniconda2/envs/phd37/bin/python'
script_path = '/home/users/gmpp/phdscripts/phdlib/emd_analysis/run_emd.py'

subprocess.call(['bsub',
                 '-o', logs_path + '%J.out',
                 '-e', logs_path + '%J.err',
                 '-W', '48:00',
                 '-R', 'rusage[mem=60000]',
                 '-M', '120000',
                 '-n', '100',
                 '-q', 'par-multi',
                 python, script_path])
