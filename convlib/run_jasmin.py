import subprocess

if __name__ == '__main__':
    logs_path = '/home/users/gmpp/logs/'
    python = '/home/users/gmpp/miniconda2/envs/phd37/bin/python'
    script_path = '/home/users/gmpp/phdlib/convlib/classifier.py'
    lcs_types = ['repelling']
    years = [x for x in range(1980, 2010)]
    for year in years:
        for lcs_type in lcs_types:
             subprocess.call(['bsub',
             '-o',logs_path+'%J.out',
             '-e',logs_path+'%J.err',
             '-W','48:00',
             '-R','rusage[mem=20000]',
             '-M','20000',
             '-n','10',
             '-q','par-multi',
             python,script_path, 'jasmin', lcs_type, str(year)])