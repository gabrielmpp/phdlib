import subprocess

if __name__ == '__main__':
    logs_path = '/home/users/gmpp/logs/'
    python = '/home/users/gmpp/miniconda2/envs/phd37/bin/python'
    script_paths = ['/home/users/gmpp/phdlib/convlib/classifier.py', ]

    for script_path in script_paths:
         subprocess.call(['bsub',
         '-o',logs_path+'%J.out',
         '-e',logs_path+'%J.err',
         '-W','12:00',
         '-R','rusage[mem=20000]',
         '-M','20000',
         '-n','20',
         '-q','par-single',
         python,script_path, 'jasmin'])




