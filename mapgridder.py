from numpy import arange, array, concatenate
from pymap3d import enu2geodetic
from xarray import DataArray
from scipy.stats import binned_statistic_2d

def grid_map(data, gridsize, lat0 = -74.21, lon0 = -113.25, var = 'D', lat = 'Lat', lon = 'Lon', statistic='mean', lat_grid_centers = None, lon_grid_centers = None):
    """
    gridsize in metres
    """
    # Make latitude and longitude grids
    (lat1,lon1) = enu2geodetic(gridsize,gridsize,0,lat0, lon0, 0)[0:2]
    
    step_lat = lat1-lat0
    step_lon = lon1-lon0

    if lat_grid_centers is None:
        lat_min = min(data[lat])
        lat_max = max(data[lat])
        lat_grid_edges = arange(lat_min, lat_max+step_lat, step_lat)
        lat_grid_centers = get_centers(lat_grid_edges)
    else:
        lat_grid_edges = get_edges(lat_grid_centers)

    if lon_grid_centers is None:      
        lon_min = min(data[lon])
        lon_max = max(data[lon])
        lon_grid_edges = arange(lon_min, lon_max+step_lon, step_lon)
        lon_grid_centers = get_centers(lon_grid_edges)
    else:
        lon_grid_edges = get_edges(lon_grid_centers)

    # Grid map
    gridded_map = binned_statistic_2d(data[lat].values, data[lon].values, data[var].values,
                                      bins=[lat_grid_edges,lon_grid_edges], 
                                      statistic=statistic)[0]
    return DataArray(gridded_map, 
                     coords = {'Lat': lat_grid_centers,
                               'Lon': lon_grid_centers}
                    )

def get_edges(centers):
    centers = array(centers)
    mid = centers[:-1] + (centers[1:]-centers[:-1])/2
    first = centers[0] - (centers[1]-centers[0])/2
    last  = centers[-1] + (centers[-1]-centers[-2])/2
    return concatenate([[first], mid, [last]])

def get_centers(edges):
    edges = array(edges)
    centers = (edges[:-1]+edges[1:])/2
    return centers