import subprocess

if __name__ == '__main__':
    logs_path = '/home/users/gmpp/logs/'
    python = '/home/users/gmpp/miniconda2/envs/phd37/bin/python'
    script_path = '/home/users/gmpp/phdlib/convlib/classifier.py'
    lcs_time_lens = [8, 16]
    years = [x for x in range(1980, 2010)]
    for year in years:
        for lcs_time_len in lcs_time_lens:
            subprocess.call(['bsub',
                             '-o', logs_path + '%J.out',
                             '-e', logs_path + '%J.err',
                             '-W', '168:00',
                             '-R', 'rusage[mem=20000]',
                             '-M', '50000',
                             #'-n', '10',
                             '-q', 'long-serial',
                             python, script_path, 'jasmin', 'repelling', str(year), str(lcs_time_len)])
