from numpy import arange
from pymap3d import enu2geodetic
from xarray import DataArray
from scipy.stats import binned_statistic_2d

def grid_map(data, gridsize, lat0 = -74.21, lon0 = -113.25, var = 'D', lat = 'Lat', lon = 'Lon', statistic='mean'):
    """
    gridsize in metres
    """
    # Make latitude and longitude grids
    (lat1,lon1) = enu2geodetic(gridsize,gridsize,0,lat0, lon0, 0)[0:2]
    
    step_lat = lat1-lat0
    step_lon = lon1-lon0
    
    lat_min = min(data[lat])
    lat_max = max(data[lat])
    lon_min = min(data[lon])
    lon_max = max(data[lon])
    
    lat_grid_edges = arange(lat_min, lat_max+step_lat, step_lat)
    lon_grid_edges = arange(lon_min, lon_max+step_lon, step_lon)

    # Grid map
    gridded_map = binned_statistic_2d(data[lat].values, data[lon].values, data[var].values,
                                      bins=[lat_grid_edges,lon_grid_edges], 
                                      statistic=statistic)[0]
    return DataArray(gridded_map, 
                     coords = {'Lat': (lat_grid_edges[:-1] + lat_grid_edges[1:])/2,
                               'Lon': (lon_grid_edges[:-1] + lon_grid_edges[1:])/2}
                    )
    