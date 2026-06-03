import numpy as np
import pandas as pd

def find_interfered_pings(ds, intensity_variable = 'intensity', 
                          beam_dim = 'beam', range_dim = 'range', 
                          intensity_threshold = 140, percentage_threshold = 10):
    """
    Find pings where more than percentage_threshold % of samples have an echo intensity 
    (averaged over all beams) that is higher than the intensity_threshold.
    """
    max_allowed_samples_above_threshold = len(ds[range_dim])*percentage_threshold/100
    above_threshold    = ds[intensity_variable].mean(dim=beam_dim) > intensity_threshold
    ping_is_interfered = above_threshold.sum(dim=range_dim) >= max_allowed_samples_above_threshold

    return ping_is_interfered

def distance_to_interface(ds, echo_intensity = 'intensity', range = 'along_beam_range',
                          time_dim = 'time', beam_dim = 'beam',
                          rmin = 200, rmax=1150, ampmin = 100, 
                          interpolate = True):
    """
    Detects interface based on echo intensity data. 
    
    Parameters:
        ds   : xarray dataset with ADCP echo intensity
        rmin : Cells closer than rmin from the sensor will be ignored
        rmax : Cells further than rmax from the sensor will be ignored
        interpolate : Subcell resolution using quadratic interpolation
    """
    
    ok_cells = (ds[range]>rmin) & (ds[echo_intensity]>ampmin)
    if rmax is not None:
        ok_cells = ok_cells & (ds[range]<rmax)
    amp = ds[echo_intensity].where(ok_cells, 0)
    
    range_dim = ds[range].dims[0]
    interface_detection_index = amp.argmax(dim=range_dim)
    distance = ds[range][interface_detection_index].rename('distance_to_interface')
    
    if interpolate:
        # Ensure constant cell size
        cell_size = ds[range].diff(dim=range_dim).mean().values
        if ds[range].diff(dim=range_dim).std() > cell_size * 1e-3:
            raise ValueError('Cell size varies, interpolation not possible. Use interpolate = False in range_to_surface call.')
        
        # Quadratic interpolation
        adjustment = refine_peaks(ds[echo_intensity], range_dim, interface_detection_index)
        distance = distance + adjustment * cell_size
    
    # Discard pings where no cells fulfilled the threshold values
    ok_range = ok_cells.any(dim=range_dim)
    distance = distance.where(ok_range, np.nan)
    return distance

def refine_peaks(values, dim, index_peak):
    """
    Refine position of peak to aquire sub-cell resolution, using a 3-point quadratic
    interpolation (see equation 1 in Wahlgren et al 2026). This function returns an 
    adjustment value between -0.5 and 0.5, and the refined peak location is at the 
    detected peak + the adjustment value times the grid size.
    
    Assumes constant grid spacing of the dimension to be interpolated over.
    """
    
    # Index right before the peak:
    # If peak is detected in the first element, this is set to same as index of the peak
    index_n1 = (index_peak - 1).where(index_peak>=1,0)
    
    # Index right after the peak:
    # If peak is detected in the last element, this is set to same as index of the peak
    N = values.sizes[dim]
    index_p1 = (index_peak + 1).where(index_peak<=N-2,N-1)
    
    # Values of the three points around the peak:
    a_n1 = values.isel({dim : index_n1})
    a_0  = values.isel({dim : index_peak})  
    a_p1 = values.isel({dim : index_p1})
    
    adjustment = (a_n1-a_p1)/(2*(a_n1-2*a_0+a_p1))
    
    # Set adjustment to 0 if not a true peak
    is_true_peak = (a_0 > a_n1) & (a_0 > a_p1)
    adjustment = adjustment.where(is_true_peak,0)

    return adjustment