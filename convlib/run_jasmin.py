import subprocess

if __name__ == '__main__':
    logs_path = '/home/users/gmpp/logs/'
    python = '/home/users/gmpp/miniconda2/envs/phd37/bin/python'
    script_path = '/home/users/gmpp/phdscripts/phdlib/convlib/classifier.py'
    lcs_time_lens = [20, 24, 28, 32]
    for lcs_time_len in lcs_time_lens:
        subprocess.call(['bsub',
                                 '-o', logs_path + '%J.out',
                                 '-e', logs_path + '%J.err',
                                 '-W', '48:00',
                                 '-R', 'rusage[mem=60000]',
                                 '-M', '120000',
                                 '-n', '40',
                                 '-q', 'par-multi',
                                 python, script_path, str(lcs_time_len)]
                        )
