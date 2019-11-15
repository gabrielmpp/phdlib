import subprocess

if __name__ == '__main__':
    logs_path = '/home/users/gmpp/logs/'
    python = '/home/users/gmpp/miniconda2/envs/phd37/bin/python'
    script_path = '/home/users/gmpp/phdlib/convlib/diagnostics.py'
    lcs_time_lens = ['8']
    seasons = ['DJF', 'JJA']
    CZs = ['0', '1']
    basins = ['Uruguai', 'Tiete']

    for basin in basins:
        for lcs_time_len in lcs_time_lens:
            for CZ in CZs:
                for season in seasons:
                    subprocess.call(['bsub',
                                     '-o', logs_path + '%J.out',
                                     '-e', logs_path + '%J.err',
                                     '-W', '12:00',
                                     '-R', 'rusage[mem=60000]',
                                     '-M', '20000',
                                     #'-n', '10',
                                     '-q', 'short-serial',
                                     python, script_path, lcs_time_len, basin, season, CZ])
