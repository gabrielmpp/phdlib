import subprocess

if __name__ == '__main__':
    logs_path = '/home/users/gmpp/logs/'
    python = '/home/users/gmpp/miniconda2/envs/phd37/bin/python'
    script_path = '/home/users/gmpp/phdlib/convlib/classifier2.py'
    lcs_time_lens = [6, 8]
    years = [x for x in range(1980, 2010)]
    for year in years:
        for lcs_time_len in lcs_time_lens:
            subprocess.call(['bsub',
                             '-o', logs_path + '%J.out',
                             '-e', logs_path + '%J.err',
                             '-W', '48:00',
                             '-R', 'rusage[mem=30000]',
                             '-M', '30000',
                             '-n', '5',
                             '-q', 'par-single',
                             python, script_path, 'jasmin', 'repelling', str(year), str(lcs_time_len)])