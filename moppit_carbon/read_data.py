import h5py

files_path='/group_workspaces/jasmin4/upscale/gmpp/carbon/'

def extract_data(file_path):
    """

    :param file_path: str
    :return: 3D array (360,180,2)
    """
    variable = 'RetrievedCOTotalColumnDiagnosticsDay'
    f = h5py.File(file_path, 'r')
    data = f['HDFEOS']['GRIDS']['MOP03']['Data Fields'][variable]
    return data.value