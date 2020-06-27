import subprocess
import uuid
import datetime
import os

config_jasmin = {
    'data_basepath': '/media/gabriel/gab_hd/data/sample_data/',
    'u_filename': 'viwve_ERA5_6hr_{year}010100-{year}123118.nc',
    'v_filename': 'viwvn_ERA5_6hr_{year}010100-{year}123118.nc',
    'tcwv_filename': 'tcwv_ERA5_6hr_{year}010100-{year}123118.nc',
    'time_freq': '6H',
    'chunks': {
        'time': 100, }
        ,
    'array_slice': {'time': slice('2000-02-06T00:00:00', '2000-02-07T18:00:00'),
                   'latitude': slice(-40, -20),
                   'longitude': slice(-50, -30),
                   # 'latitude': slice(-20, -35),
                   # 'longitude': slice(-55, -35)
                    },

    'array_slice_latlon': {
        'latitude': slice(-70, 35),
        'longitude': slice(-155, 5),
        # 'latitude': slice(-20, -35),
        # 'longitude': slice(-55, -35)
    },
    'array_slice_time': {
        'time': slice(None, None),
    }
    }

if __name__ == '__main__':
    logs_path = '/home/users/gmpp/logs/'
    python = '/home/users/gmpp/miniconda2/envs/phd37/bin/python'
    script_path = '/home/users/gmpp/phdscripts/phdlib/convlib/classifier.py'
    lcs_time_lens = [4, 8, 12, 16]
    years = range(2000, 2009)
    start_year = 1982
    end_year = 2009
    config = config_jasmin
    config['start_year'] = start_year
    config['end_year'] = end_year
    config['array_slice_time']['time'] = slice(f'{start_year}-01-01T00:00:00', f'{end_year}-12-31T18:00:00')
    config['start_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config['data_basepath'] = '/gws/nopw/j04/primavera1/observations/ERA5/'
    # outpath_temp = '/group_workspaces/jasmin4/upscale/gmpp/convzones/experiment_{}/'.format(uuid.uuid4())

    for lcs_time_len in lcs_time_lens:
        config['lcs_time_len'] = lcs_time_len
        outpath_temp = '/work/scratch-pw/gmpp/experiment_timelen_{timelen}_{id}/'.format(id=uuid.uuid4(),
                                                                                         timelen=str(lcs_time_len),
                                                                                         end_year=str(end_year))
        os.mkdir(outpath_temp)
        with open(outpath_temp + 'config.txt', 'w') as f:
            f.write(str(config))
        # ---- Preprocess data here ---- #
        # Crop input data appropriately considering last seqs from previous year.
        for year in years:
            subprocess.call(['bsub',
                             '-o', logs_path + '%J.out',
                             '-e', logs_path + '%J.err',
                             '-W', '48:00',
                             '-R', 'rusage[mem=50000]',
                             '-M', '150000',
                             # '-m', 'skylake348G',
                             '-n', '15',
                             '-q', 'par-single',
                             python, script_path, str(lcs_time_len), str(year)]
                            )