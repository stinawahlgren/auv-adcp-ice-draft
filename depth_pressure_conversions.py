from pandas import read_csv, DataFrame, concat
from xarray import DataArray
import numpy as np
from glob import glob
from gsw import geo_strf_dyn_height, z_from_p, p_from_z
from scipy.stats import binned_statistic

def compute_dynamic_height_profile(ctd_files = 'data/auxiliary/CTD/CTD_NBP2202_*.txt', 
                                   SA = 'AbsSal (g/kg)', CT =  'ConsTemp (deg)', p = 'Pressure (dB)'):
    """
    Computes gesostrophic dynamical height (referenced at p=0) as a function of pressure, based on 
    mean profiles of  conservative temperature and absolute salinity. Data is taken from the specified ctd-files

    Parameters:

    ctd_files : file names of ctd data
    SA       : name of column in ctd_data with absolute salinity (unit: g/kg)
    CT       : name of column in ctd_data with conservative temperature (unit: degC)
    p        : name of column in ctd_data with pressure (unit: dBar)

    Returns:

    xarray.DataArray with dynamical height as a function of pressure.
    """
    ctd_data = read_ctd_data(ctd_files)
    return dyn_height_profile(ctd_data, SA = SA, CT = CT, p = p)


def depth_from_pressure(p, lat=-74, dynamic_height_profile = None, p_dim = 'p'):
    """
    Converts from pressure to depth using gsw.z_from_p.
    The dynamic_height_profile is given as an xarray.DataArray of dynamic height as a function of pressure. 
    The name of the pressure dimension is defined by p_dim.

    If dymanic_height_profile  is not provided, a dynamic height of 0 will be used.
    """
    if dynamic_height_profile is None:
        dynamic_height = 0
    else:
        dynamic_height = dynamic_height_profile.interp({p_dim : p}).values

    return -z_from_p(p, lat, dynamic_height)


def pressure_from_depth(depth, lat=-74, dynamic_height_profile = None, p_dim = 'p'):
    """
    Converts from depth to pressure using gsw.z_from_p.
    The dynamic_height_profile is given as an xarray.DataArray of dynamic height as a function of pressure. 
    The name of the pressure dimension is defined by p_dim.

    If dymanic_height_profile is not provided, a dynamic height of 0 will be used.
    """
    if dynamic_height_profile is None:
        dynamic_height = 0
    else:
        p = p_from_z(-depth, lat, 0)
        dynamic_height = dynamic_height_profile.interp({p_dim : p}).values

    return p_from_z(-depth, lat, dynamic_height)


def read_ctd_data(filename = 'data/auxiliary/CTD/CTD_NBP2202_*.txt'):
    """
    Load provided csv-files as a pandas.DataFrame
    """
    ctd_files = list_of_files(filename)
    ctd_data_list = [read_csv(f) for f in ctd_files]
    return concat(ctd_data_list, axis=0, ignore_index = True)


def mean_profile(data, var, p, bins):
    """
    Computes mean profile
    """
    res = binned_statistic(data[p].values,
                           data[var].values,
                           statistic='mean', 
                           bins= bins)
    values = res.statistic
    ps = (res.bin_edges[:-1] + res.bin_edges[1:])/2
    return (values, ps)


def dyn_height_profile(ctd_data, SA = 'AbsSal (g/kg)', CT =  'ConsTemp (deg)', p = 'Pressure (dB)'):
    """
    Computes gesostrophic dynamical height (referenced at p=0) as a function of pressure, based on 
    mean profiles of  conservative temperature and absolute salinity.

    Parameters:

    ctd_data : pandas.DataFrame with CTD data
    SA       : name of column in ctd_data with absolute salinity (unit: g/kg)
    CT       : name of column in ctd_data with conservative temperature (unit: degC)
    p        : name of column in ctd_data with pressure (unit: dBar)

    Returns:

    xarray.DataArray with dynamical height as a function of pressure.
    """
    
    # Compute mean profiles of conservative temperature and absoulte salinity
    bin_size = 1
    bins = np.arange(5,np.max(ctd_data['Pressure (dB)']),bin_size)
    (SA,p) = mean_profile(ctd_data, 'AbsSal (g/kg)', 'Pressure (dB)', bins)
    (CT,p) = mean_profile(ctd_data, 'ConsTemp (deg)', 'Pressure (dB)', bins)

    # Replace nan with lineraly interpolated values
    df = DataFrame({'p' : p,
                    'SA':SA,
                    'CT':CT})
    df = df.interpolate()

    # Compute dynamic height
    dyn_height = geo_strf_dyn_height(df['SA'], df['CT'], df['p'], p_ref = 0)

    return DataArray(data = dyn_height, dims = 'p', coords = {'p':p})


def list_of_files(filenames):
    """
    Returns a list of files matching the input

    Example usage:
    >>> list_of_files('path/to/files/*.csv')
    ['path/to/files/file1.csv', 'path/to/files/file2.csv']
    
    >>> list_of_files(['path/to/files/*.csv', 'otherfile.txt'])
    ['path/to/files/file1.csv', 'path/to/files/file2.csv', 'otherfile.txt']
    """
    if type(filenames) in (list,tuple):
        file_list = []
        for filename in filenames:
            for f in glob(filename):
                file_list.append(f)
    else:
        file_list = glob(filenames)
    return file_list